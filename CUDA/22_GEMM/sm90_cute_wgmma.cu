#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/device_kernel.h"
#include "utils.h"

using namespace cute;

__host__ __device__ void
print1(half_t v)
{
    printf("%*.2f ", 2, float(v));
}

__device__ int canonical_warp_idx_sync()
{
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

// nvcc cute_wgmma.cu -O3 -arch=sm_90a -I ../../cutlass-4.1/include/ -I ../../cutlass-4.1/tools/util/include/ -lcuda -lcublas -o wgmma_tma --expt-relaxed-constexpr && ./wgmma_tma
template <int bM, int bN, int bK, int NumThreads, class T, class TC, int NumPipe = 1, int base,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC,
          class TmaA, class TmaB, class TmaC>
__global__ void wgmma_tma_kernel(T *A, T *B, TC *C, int M, int N, int K,
                                 CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                 CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                 CUTLASS_GRID_CONSTANT TmaC const tma_c)
{
    alignas(128) extern __shared__ T shared_memory[];
    T *sA_ptr = shared_memory;
    T *sB_ptr = shared_memory + bM * bK * NumPipe;

    // make smem tensors
    Tensor sA = make_tensor(make_smem_ptr(sA_ptr), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr), SmemLayoutB{});
    Tensor sC = make_tensor(make_smem_ptr<TC>(shared_memory), SmemLayoutC{});

    // make tiled_mma
    TiledMMA mma = make_tiled_mma(SM90_64x128x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::MN>{});

    // make tma coord tensor
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
    Tensor mC = tma_c.get_tma_tensor(make_shape(M, N));

    int bx = blockIdx.x;
    int by = blockIdx.y;
    by = (by << base) + (bx & ((1 << base) - 1));
    bx = (bx >> base);

    // cute block gmem tensor
    Tensor gA = local_tile(mA, make_shape(Int<bM>{}, Int<bK>{}), make_coord(bx, _));
    Tensor gB = local_tile(mB, make_shape(Int<bN>{}, Int<bK>{}), make_coord(by, _));
    Tensor gC = local_tile(mC, make_shape(Int<bM>{}, Int<bN>{}), make_coord(bx, by));

    // tma partition
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));
    auto [tCgC, tCsC] = tma_partition(tma_c, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sC), group_modes<0, 2>(gC));

    // tma expect-tx bytes
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    // init mbarrier
    __shared__ alignas(8) uint64_t producer_mbar[NumPipe];
    __shared__ alignas(8) uint64_t consumer_mbar[NumPipe];

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    CUTE_UNROLL
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        if (warp_idx == 0 && lane_predicate == 1)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    __syncthreads();

    // tma prefetch
    CUTE_UNROLL
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        if (warp_idx == 0 && lane_predicate == 1)
        {
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        ++k_tile;
    }

    // make wgmma thread tensor
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsCp = thr_mma.partition_C(sC);
    Tensor tCrC = thr_mma.make_fragment_C(tCsCp);
    clear(tCrC);

    Copy_Atom<SM90_U32x4_STSM_N, half_t> s2r_atom;
    TiledCopy s2r_copy = make_tiled_copy_C(s2r_atom, mma);
    ThrCopy s2r_thr_copy = s2r_copy.get_slice(threadIdx.x);
    Tensor tXrC = s2r_thr_copy.retile_S(tCrC);
    Tensor tXsC = s2r_thr_copy.partition_D(sC);

    // get stages for mbarrier
    auto write_state = cutlass::PipelineState<NumPipe>();
    auto read_state = cutlass::PipelineState<NumPipe>();

    for (int pipe = 0; pipe < NumPipe - 1; ++pipe)
    {
        ProducerBarType::wait(&producer_mbar[pipe], read_state.phase());
        ++read_state;

        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, pipe), tCrB(_, _, _, pipe), tCrC);
        warpgroup_commit_batch();
        ConsumerBarType::arrive(&consumer_mbar[pipe]);
    }

    while (k_tile <= k_tile_count)
    {
        warpgroup_wait<1>();
        if (warp_idx == 0 && lane_predicate == 1 && k_tile < k_tile_count)
        {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;
        ++k_tile;
    }
    warpgroup_wait<0>();

    copy(s2r_atom, tXrC, tXsC);

    tma_store_fence();
    __syncthreads();

    // store shared memory to global memory
    if (warp_idx == 0 && lane_predicate == 1)
    {
        copy(tma_c, tCsC, tCgC);
        tma_store_arrive();
        tma_store_wait<0>();
    }
}

// A, B, and C are device pointers
extern "C" void solve(half_t *A, half_t *B, half_t *C, int M, int N, int K, float alpha, float beta)
{

    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 64;
    constexpr int numPipe = 3;

    using T = half_t;
    using TC = half_t;
    using namespace cute;

    constexpr int num_threads = 128; // one warpgroup
    int num_blockM = (M + blockM - 1) / blockM;
    int num_blockN = (N + blockN - 1) / blockN;

    constexpr int base = 0;
    num_blockM = num_blockM * (1 << base);
    num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

    auto smemA_layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<blockM>{}, Int<blockK>{}, Int<numPipe>{}), Step<_1, _2, _3>{});  // load to smemA
    auto smemB_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(Int<blockN>{}, Int<blockK>{}, Int<numPipe>{}), Step<_2, _1, _3>{}); // load to smemB
    auto smemC_layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<TC>{}, make_shape(Int<blockM>{}, Int<blockN>{}));                                     // store from smemC

    // make cute gmem tensor
    Tensor tensorA = make_tensor(A, make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor tensorB = make_tensor(B, make_shape(N, K), make_stride(Int<1>{}, N));
    Tensor tensorC = make_tensor(C, make_shape(M, K), make_stride(N, Int<1>{}));

    // make tma tiled_copy
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, tensorA, smemA_layout(_, _, 0), make_shape(Int<blockM>{}, Int<blockK>{}));
    auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, tensorB, smemB_layout(_, _, 0), make_shape(Int<blockN>{}, Int<blockK>{}));
    auto tma_c = make_tma_atom(SM90_TMA_STORE{}, tensorC, smemC_layout, make_shape(Int<blockM>{}, Int<blockN>{}));

    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    auto kernel_fptr = wgmma_tma_kernel<blockM, blockN, blockK, num_threads, T, TC, numPipe, base,
                                        decltype(smemA_layout), decltype(smemB_layout), decltype(smemC_layout),
                                        decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K, tma_a, tma_b, tma_c);
}

// nvcc sm90_cute_wgmma.cu -O3 -arch=sm_90a -I ../../cutlass-4.1/include/ -I ../../cutlass-4.1/tools/util/include/ -lcuda -lcublas -o wgmma_tma --expt-relaxed-constexpr && ./wgmma_tma
// cublas time = 0.178407 ms, TFLPOS = 770.367145, mfu = 0.778935
// mma time = 0.189971 ms, TFLPOS = 723.473953, mfu = 0.731521
int main()
{
    srand(1234);

    int num = 4096;

    int M = num;
    int N = num;
    int K = num;

    using T = half_t;  // A, B dtype
    using TC = half_t; // C dtype

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);
    thrust::host_vector<TC> mma_res(M * N);
    thrust::host_vector<TC> ref_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_C[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<TC> d_C1 = h_C;
    thrust::device_vector<TC> d_C2 = h_C;

    solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, 1.0f, 0.0f);
    mma_res = d_C2;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                 reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                 &beta,
                 reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    ref_res = d_C1;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 989e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                         reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                         &beta,
                         reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                         CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, 1.0f, 0.0f);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}
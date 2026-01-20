#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include "utils.h"

using namespace cute;

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void mma_kernel(T *A, T *B, T *C, int M, int N, int K, float alpha, float beta)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int y = (by << base) + (bx & ((1 << base) - 1));
    int x = (bx >> base);

    const int tid = threadIdx.x;

    // make gmem cute tensors, A, B, C are all k-major
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{})); // row-major
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(Int<1>{}, N)); // row-major
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{})); // row-major

    // get block gmem tensors
    Tensor gA = local_tile(mA, make_shape(Int<bM>{}, Int<bK>{}), make_coord(x, _));
    Tensor gB = local_tile(mB, make_shape(Int<bN>{}, Int<bK>{}), make_coord(y, _));
    Tensor gC = local_tile(mC, make_shape(Int<bM>{}, Int<bN>{}), make_coord(x, y));

    // auto swizzle_atom = Layout<Shape<_128, _64>, Stride<_64, _1>>{}; // none swizzle
    auto swizzle_atom_a = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}); // 128B swizzle atom K-major
    auto swizzle_atom_b = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}); // 128B swizzle atom N-major

    // make smem layout
    auto smem_layout_a = tile_to_shape(swizzle_atom_a, make_shape(Int<bM>{}, Int<bK>{}, Int<NumPipe>{})); // Step<>?
    auto smem_layout_b = tile_to_shape(swizzle_atom_b, make_shape(Int<bN>{}, Int<bK>{}, Int<NumPipe>{}));

    extern __shared__ T shared_memory[];
    T *sA_ptr = shared_memory;
    T *sB_ptr = shared_memory + bM * bK * NumPipe;

    // make smem tensors
    Tensor sA = make_tensor(make_smem_ptr(sA_ptr), smem_layout_a);
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr), smem_layout_b);
    Tensor sC = make_tensor(make_smem_ptr(shared_memory), Layout<Shape<Int<bM>, Int<bN>>, Stride<Int<bN>, _1>>{});

    // make copy and mma atom
    using copy_g2s = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>; // cp.async
    using copy_s2r_a = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;                  // ldmatrix.x4
    using copy_s2r_b = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;                  // ldmatrix.x4.trans
    using mma_atom = SM80_16x8x16_F32F16F16F32_TN;                            // mma.sync

    using copy_thr_layout = Layout<Shape<_16, _8>, Stride<_8, _1>>; // 128 threads
    using copy_val_layout = Layout<Shape<_1, _8>>;                  // 1 thread 8 values

    // make tiled copy to copy tensor from gmem to smem
    TiledCopy tiled_copy_g2s_a = make_tiled_copy(copy_g2s{}, copy_thr_layout{}, copy_val_layout{});
    TiledCopy tiled_copy_g2s_b = make_tiled_copy(copy_g2s{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}, Layout<Shape<_8, _1>>{});

    // make tiledMMA
    TiledMMA tiled_mma = make_tiled_mma(mma_atom{}, Layout<Shape<_2, _2>>{}, Tile<_32, _32, _16>{});

    // make tiled copy to copy tensor from smem to rmem using ldmatrix
    TiledCopy tiled_copy_s2r_a = make_tiled_copy_A(copy_s2r_a{}, tiled_mma);
    TiledCopy tiled_copy_s2r_b = make_tiled_copy_B(copy_s2r_b{}, tiled_mma);

    ThrCopy thr_copy_g2s_a = tiled_copy_g2s_a.get_slice(tid);
    Tensor tAgA = thr_copy_g2s_a.partition_S(gA);
    Tensor tAsA = thr_copy_g2s_a.partition_D(sA);

    ThrCopy thr_copy_g2s_b = tiled_copy_g2s_b.get_slice(tid);
    Tensor tBgB = thr_copy_g2s_b.partition_S(gB);
    Tensor tBsB = thr_copy_g2s_b.partition_D(sB);

    // prefetch
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0;

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
    {
        copy(tiled_copy_g2s_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(tiled_copy_g2s_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    ThrMMA thr_mma = tiled_mma.get_slice(tid);
    Tensor tCgC = thr_mma.partition_C(gC);

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    ThrCopy thr_copy_s2r_a = tiled_copy_s2r_a.get_slice(tid);
    Tensor tXsA = thr_copy_s2r_a.partition_S(sA);
    Tensor tXrA = thr_copy_s2r_a.retile_D(tCrA);

    ThrCopy thr_copy_s2r_b = tiled_copy_s2r_b.get_slice(tid);
    Tensor tXsB = thr_copy_s2r_b.partition_S(sB);
    Tensor tXrB = thr_copy_s2r_b.retile_D(tCrB);

    // STSM needs sm_90
    // Copy_Atom<SM90_U32x4_STSM_N, half_t> r2s_atom;
    // TiledCopy r2s_copy = make_tiled_copy_C(r2s_atom, tiled_mma);
    // ThrCopy r2s_thr_copy = r2s_copy.get_slice(threadIdx.x);
    // Tensor tXrC = r2s_thr_copy.retile_S(tCrC);
    // Tensor tXsC = r2s_thr_copy.partition_D(sC);

    int smem_pipe_read = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Pipe slice
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1)
    {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tiled_copy_s2r_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        copy(tiled_copy_s2r_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Slice the smem_pipe_read smem
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
            copy(tiled_copy_s2r_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(tiled_copy_s2r_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(tiled_copy_g2s_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(tiled_copy_g2s_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
            }
            // Thread-level register gemm for k_block
            gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    //     copy(r2s_atom, tXrC, tXsC); // stmatrix save to smem
    //     __syncthreads();

    //     // store from smem to gmem
    //     int n_tid_col = bN / 8; // use float4
    //     auto C_tile = C + x * bM * N + y * bN;
    // #pragma unroll
    //     for (int i = tid; i < bM * n_tid_col; i += NumThreads)
    //     {
    //         int row = i / n_tid_col;
    //         int col = i % n_tid_col;
    //         reinterpret_cast<float4 *>(C_tile + row * N)[col] = reinterpret_cast<float4 *>(shared_memory + row * bN)[col];
    //     }

    axpby(alpha, tCrC, beta, tCgC); // TODO: use stmatrix
}

// A, B, and C are device pointers
extern "C" void solve(half_t *A, half_t *B, half_t *C, int M, int N, int K, float alpha, float beta)
{
    using T = half_t;
    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 64;
    constexpr int numPipe = 3;

    constexpr int num_threads = 128;
    int num_blockM = (M + blockM - 1) / blockM;
    int num_blockN = (N + blockN - 1) / blockN;

    constexpr int base = 0;
    num_blockM = num_blockM * (1 << base);
    num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    auto kernel_fptr = mma_kernel<blockM, blockN, blockK, num_threads, T, numPipe, base>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta);
}

// nvcc cute_mma.cu -O3 -arch=sm_80 -I ../../cutlass-3.8/include/ -I ../../cutlass-3.8/tools/util/include/ -lcuda -lcublas -o mma --expt-relaxed-constexpr && ./mma
int main()
{
    srand(1234);

    // A row-major
    // B row-major

    int num = 4096;
    int M = num;
    int N = num;
    int K = num;

    using T = half_t;

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<T> h_C(M * N);
    thrust::host_vector<T> mma_res(M * N);
    thrust::host_vector<T> ref_res(M * N);

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
    thrust::device_vector<T> d_C1 = h_C;
    thrust::device_vector<T> d_C2 = h_C;

    solve(d_A.data().get(), d_B.data().get(), d_C1.data().get(), M, N, K, 1.0f, 1.0f);
    mma_res = d_C1;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 1.0f;
    // C is column-major
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                 reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                 &beta,
                 reinterpret_cast<__half *>(d_C2.data().get()), CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    ref_res = d_C2;

    // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
    //         reinterpret_cast<const __half *>(&alpha),
    //         reinterpret_cast<__half *>(d_A.data().get()), K,
    //         reinterpret_cast<__half *>(d_B.data().get()), N,
    //         reinterpret_cast<const __half *>(&beta),
    //         reinterpret_cast<__half *>(d_C2.data().get()), M);

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 312e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                         reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                         &beta,
                         reinterpret_cast<__half *>(d_C2.data().get()), CUDA_R_16F, N,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C1.data().get(), M, N, K, 1.0f, 1.0f);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}
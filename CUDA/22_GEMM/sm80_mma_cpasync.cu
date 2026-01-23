#include <iostream>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <random>
#include "utils.h"

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(void const *const ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ static void
mma(float *d, uint32_t *a, uint32_t *b, float *c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    if constexpr (N == 0)
    {
        asm volatile("cp.async.wait_all;\n" ::);
    }
    else
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
}

template <class T>
__device__ __forceinline__ void cp_async_size16(T *smem, const T *gmem)
{
    __uint32_t smem_int_ptr = static_cast<__uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                 "l"(gmem),
                 "n"(sizeof(float4)));
}

template <class T>
__device__ __forceinline__ static void
ldmatrix_x4(T *smem_src, uint32_t *dst)
{
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "r"(smem_int_ptr));
}

template <class T>
__device__ __forceinline__ static void
ldmatrix_x4_trans(T *smem_src, uint32_t *dst)
{
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "r"(smem_int_ptr));
}

template <class T>
__device__ __forceinline__ static void
ldmatrix_copy_a(T *tile_sA, uint32_t *reg_a, int lane_id, int warp_row, int k_step)
{
    int row = (lane_id % 16) + warp_row * 16;
    int col = k_step * 2 + (lane_id / 16);
    int swizzle_col = (row % 8) ^ col;
    int offset_base = row * 64 + swizzle_col * 8;
    int reg_idx = (k_step & 1) * 16;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldmatrix_x4(tile_sA + offset_base + i * 2048, reg_a + reg_idx + i * 4); // i + 1, row + 32
    }
}

template <class T>
__device__ __forceinline__ static void
ldmatrix_copy_b(T *tile_sB, uint32_t *reg_b, int lane_id, int warp_col, int k_step)
{
    // warp0 warp2 warp0 warp2
    int row = (lane_id % 16) + k_step * 16;
    int col_base = (lane_id / 16) * 2 + warp_col;
    int reg_idx = (k_step & 1) * 16;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int col = col_base + i * 4;
        int swizzle_col = (row % 8) ^ col;
        int offset = row * 128 + swizzle_col * 8;
        ldmatrix_x4_trans(tile_sB + offset, reg_b + reg_idx + i * 4);
    }
}

__device__ __forceinline__ static void
mma_sync(int M, int N, uint32_t *a, uint32_t *b, float *c, int stage)
{
    int offset_idx = (stage & 1) * 16;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int n = 0; n < N; ++n)
        {
            auto c_offset = c + (m * 8 + n) * 4;
            auto a_offset = a + offset_idx + m * 4;
            auto b_offset = b + offset_idx + n * 2;
            mma(c_offset, a_offset, b_offset, c_offset);
        }
    }
}

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void mma_cpasync_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C, int M, int N, int K, float alpha, float beta)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_row = warp_id % 2;
    const int warp_col = warp_id / 2;

    // thread block swizzle
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int y = (oy << base) + (ox & ((1 << base) - 1));
    int x = (ox >> base);

    auto gA = A + x * bM * K;          // A is K-major
    auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    alignas(8) extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = shared_memory + bM * bK * NumPipe;

    float reg_c[128] = {0}; // [4][8][4] float
    uint32_t reg_a[32];     // [2][4][4]
    uint32_t reg_b[32];     // [2][8][2]

    constexpr int m_size = 4;                            // 128 / 16 / 2
    constexpr int n_size = 8;                            // 128 / 8 / 2
    constexpr int nbK = bK / sizeof(float4) * sizeof(T); // K-major 64 / 8 = 8
    constexpr int nbN = bN / sizeof(float4) * sizeof(T); // N-major 128 / 8 = 16

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;

    int cpasync_a_offset_row = tid / nbK;
    int cpasync_a_offset_col = tid % nbK;
    int cpasync_a_offset_col_sw = (cpasync_a_offset_row % 8) ^ cpasync_a_offset_col;
    int cpasync_a_smem_offset = cpasync_a_offset_row * bK + cpasync_a_offset_col_sw * 8;
    int cpasync_a_gmem_offset = cpasync_a_offset_row * K + cpasync_a_offset_col * 8;

    int cpasync_b_offset_row = tid / nbN;
    int cpasync_b_offset_col = tid % nbN;
    int cpasync_b_offset_col_sw = (cpasync_b_offset_row % 8) ^ cpasync_b_offset_col;
    int cpasync_b_smem_offset = cpasync_b_offset_row * bN + cpasync_b_offset_col_sw * 8;
    int cpasync_b_gmem_offset = cpasync_b_offset_row * N + cpasync_b_offset_col * 8;

    auto cp_async_load = [&](T *tile_sA, const T *tile_gA, T *tile_sB, const T *tile_gB)
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int smem_offset = cpasync_a_smem_offset + i * 16 * bK;
            int gmem_offset = cpasync_a_gmem_offset + i * 16 * K;
            cp_async_size16(tile_sA + smem_offset, tile_gA + gmem_offset);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int smem_offset = cpasync_b_smem_offset + i * 8 * bN;
            int gmem_offset = cpasync_b_gmem_offset + i * 8 * N;
            cp_async_size16(tile_sB + smem_offset, tile_gB + gmem_offset);
        }
    };

    // prefetch
#pragma unroll
    for (int k_pipe = 0; k_pipe < NumPipe - 1; ++k_pipe)
    {
        auto tile_gA = gA + k_tile_next * bK;
        auto tile_gB = gB + k_tile_next * bK * N;
        auto tile_sA = sA + k_pipe * bM * bK;
        auto tile_sB = sB + k_pipe * bN * bK;
        cp_async_load(tile_sA, tile_gA, tile_sB, tile_gB);
        asm volatile("cp.async.commit_group;\n" ::);

        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    int smem_pipe_read = 0;
    int smem_pipe_write = NumPipe - 1;

    auto tile_sA = sA + smem_pipe_read * bM * bK;
    auto tile_sB = sB + smem_pipe_read * bN * bK;

    cp_async_wait<NumPipe - 2>();
    __syncthreads();

    // prefetch smem to rmem
    ldmatrix_copy_a(tile_sA, reg_a, lane_id, warp_row, 0);
    ldmatrix_copy_b(tile_sB, reg_b, lane_id, warp_col, 0);

    constexpr int block_bK = bK / 16;

    while (k_tile_count > -(NumPipe - 1))
    {

#pragma unroll
        for (int tk = 0; tk < block_bK; ++tk)
        {
            if (tk == block_bK - 1)
            {
                tile_sA = sA + smem_pipe_read * bM * bK;
                tile_sB = sB + smem_pipe_read * bN * bK;
                cp_async_wait<NumPipe - 2>();
                __syncthreads();
            }

            int tk_next = (tk + 1) % block_bK;
            ldmatrix_copy_a(tile_sA, reg_a, lane_id, warp_row, tk_next);
            ldmatrix_copy_b(tile_sB, reg_b, lane_id, warp_col, tk_next);
            mma_sync(m_size, n_size, reg_a, reg_b, reg_c, tk);

            if (tk == 0)
            {
                auto tile_gA = gA + k_tile_next * bK;
                auto tile_sA = sA + smem_pipe_write * bM * bK;
                auto tile_gB = gB + k_tile_next * bK * N;
                auto tile_sB = sB + smem_pipe_write * bN * bK;
                cp_async_load(tile_sA, tile_gA, tile_sB, tile_gB);
                asm volatile("cp.async.commit_group;\n" ::);

                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == NumPipe - 1) ? 0 : smem_pipe_read + 1;
            }
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < m_size; ++i)
    {
        int local_row1 = lane_id / 4 + warp_row * 16 + i * 32;
        int local_row2 = local_row1 + 8;
#pragma unroll
        for (int j = 0; j < n_size; ++j)
        {
            int local_col = (lane_id % 4) * 2 + warp_col * 8 + j * 16;
            reinterpret_cast<float *>(shared_memory)[local_row1 * bN + local_col + 0] = reg_c[(i * n_size + j) * 4 + 0];
            reinterpret_cast<float *>(shared_memory)[local_row1 * bN + local_col + 1] = reg_c[(i * n_size + j) * 4 + 1];
            reinterpret_cast<float *>(shared_memory)[local_row2 * bN + local_col + 0] = reg_c[(i * n_size + j) * 4 + 2];
            reinterpret_cast<float *>(shared_memory)[local_row2 * bN + local_col + 1] = reg_c[(i * n_size + j) * 4 + 3];
        }
    }
    __syncthreads();

    int stride = bN / 4;
    int num_vectors = (bM * bN) / 4;
#pragma unroll
    for (int i = threadIdx.x; i < num_vectors; i += NumThreads)
    {
        int row = i / stride;
        int col = (i % stride) * 4;
        float4 smem_vec = reinterpret_cast<float4 *>(shared_memory)[i];

        // if use alpha and beta
        // float2 c_old = *reinterpret_cast<float2 *>(gC + row * N + col);
        // half2 *c_old_h2 = reinterpret_cast<half2 *>(&c_old);
        // half2 c_old_0 = c_old_h2[0];
        // half2 c_old_1 = c_old_h2[1];
        // smem_vec.x += static_cast<float>(c_old_0.x);
        // smem_vec.y += static_cast<float>(c_old_0.y);
        // smem_vec.z += static_cast<float>(c_old_1.x);
        // smem_vec.w += static_cast<float>(c_old_1.y);

        half2 res_h2_0 = __floats2half2_rn(smem_vec.x, smem_vec.y);
        half2 res_h2_1 = __floats2half2_rn(smem_vec.z, smem_vec.w);

        half2 res_pack[2] = {res_h2_0, res_h2_1};
        *reinterpret_cast<float2 *>(gC + row * N + col) = *reinterpret_cast<float2 *>(res_pack);
    }
}

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1>
__global__ void gemm_base(const T *A, const T *B, T *C, int M, int N, int K, float alpha, float beta)
{
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int m_tid = 16;         // 16 threads in M
    constexpr int n_tid = 8;          //  8 threads in N
    constexpr int m_val = bM / m_tid; //  8 per thread in M
    constexpr int n_val = bN / n_tid; // 16 per thread in N

    auto gA = A + x * bM * K;          // A is K-major
    auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    int row_offset = tid / n_tid * m_val;
    int col_offset = tid % n_tid * n_val;

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int row_in_block = row_offset + i;
        int global_row = x * bM + row_in_block;
        auto tA = gA + row_in_block * K;
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int col_in_block = col_offset + j;
            int global_col = y * bN + col_in_block;

            if (global_row >= M || global_col >= N)
            {
                continue;
            }
            auto tC = gC + row_in_block * N + col_in_block;
            float ori_val = static_cast<float>(tC[0]);
            float val_c = 0.0f;
            auto b_ptr = gB + col_in_block;

            for (int k = 0; k < K; ++k)
            {
                val_c += static_cast<float>(tA[k] * b_ptr[k * N]);
            }
            tC[0] = static_cast<T>(val_c * alpha + beta * ori_val);
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 64;
    constexpr int numPipe = 3;

    using T = half;
    using TC = half;

    constexpr int num_threads = 128; // one warpgroup
    int num_blockM = (M + blockM - 1) / blockM;
    int num_blockN = (N + blockN - 1) / blockN;

    constexpr int base = 0;
    num_blockM = num_blockM * (1 << base);
    num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    auto kernel_fptr = mma_cpasync_kernel<blockM, blockN, blockK, num_threads, T, numPipe, base>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta);
}

/**
 * nvcc sm80_mma_cpasync.cu -O3 -arch=sm_80 -lcuda -lcublas -o sm80_mma_cpasync && ./sm80_mma_cpasync
 * gemm success
 * cublas time = 0.568996 ms, TFLPOS = 241.546394, mfu = 0.774187
 * mma time = 0.672809 ms, TFLPOS = 204.276228, mfu = 0.654731
 */
int main()
{
    float u = 1.0;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-u, u);

    int num = 4096;

    int M = num;
    int N = num;
    int K = num;

    using T = half;  // A, B dtype
    using TC = half; // C dtype

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);
    thrust::host_vector<TC> mma_res(M * N);
    thrust::host_vector<TC> ref_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<T>(dis(gen));
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(dis(gen));
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_C[i] = static_cast<T>(dis(gen));
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<TC> d_C1 = h_C;
    thrust::device_vector<TC> d_C2 = h_C;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                 reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                 &beta,
                 reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    ref_res = d_C1;

    solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, alpha, beta);
    mma_res = d_C2;

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
                         reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, alpha, beta);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}
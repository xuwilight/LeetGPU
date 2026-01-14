#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include "utils.h"

/**
 * Block Size: 64x64x32
 * Threads: 128 (4 Warps)
 *
 * A (M×K) is row-major, B (K×N) is row-major
 *
 * 1. Block Tile: 64x64 (Computed by 1 CTA)
 * 2. Warp Tile:  32x32   (Computed by 1 Warp) -> Layout: 2x2 Warps
 * 3. Thread Tile:  4x8   (Computed by 1 Thread) -> Layout within Warp: 8x4 Threads
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v2(const T *A, const T *B, float *C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int m_val = 8; // m_val per thread in M
    const int n_val = 8; // n_val per thread in N

    // warp tile
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    // thread tile
    const int tid_row_in_warp = lane_id / 4; // 4 threads of a warp in N
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 64 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 32 + tid_col_in_warp * n_val;

    int x = blockIdx.x;
    int y = blockIdx.y;

    auto gA = A + x * bM * K;          // A is K-major
    auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    __shared__ T sA[bM * bK];
    __shared__ T sB[bN * bK];

    float reg_c[m_val * n_val] = {0.0f};
    float reg_a[m_val];
    float reg_b[n_val];

    int numK = (K + bK - 1) / bK;

    for (int k = 0; k < numK; ++k)
    {
        // copy A into sA, K-major
#pragma unroll
        for (int i = tid; i < bM * bK; i += NumThreads)
        {
            int row = i / bK;
            int col = i % bK;

            int global_row = x * bM + row;
            int global_col = k * bK + col;

            if (global_row < M && global_col < K)
            {
                sA[row * bK + col] = gA[row * K + global_col];
            }
            else
            {
                sA[row * bK + col] = 0.0f;
            }
        }

// copy B into sB, N-major
#pragma unroll
        for (int i = tid; i < bN * bK; i += NumThreads)
        {
            int row = i / bN;
            int col = i % bN;

            int global_row = k * bK + row;
            int global_col = y * bN + col;

            if (global_row < K && global_col < N)
            {
                sB[row * bN + col] = gB[global_row * N + col];
            }
            else
            {
                sB[row * bN + col] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < bK; ++tk)
        {
            // 1. Load A fragments into registers
            for (int i = 0; i < m_val; ++i)
            {
                reg_a[i] = sA[(row_offset + i) * bK + tk];
            }

            // 2. Load B fragments into registers
            for (int j = 0; j < n_val; ++j)
            {
                reg_b[j] = sB[tk * bN + col_offset + j];
            }

            // 3. Compute Outer Product
            for (int i = 0; i < m_val; ++i)
            {
                for (int j = 0; j < n_val; ++j)
                {
                    reg_c[i * n_val + j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int local_row = row_offset + i;
        int global_row = x * bM + local_row;
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int local_col = col_offset + j;
            int global_col = y * bN + local_col;
            if (global_row < M && global_col < N)
            {
                gC[local_row * N + local_col] = reg_c[i * n_val + j];
            }
        }
    }
}

__device__ __forceinline__ float ld_global(const float *ptr)
{
    float ret;
    asm volatile("ld.global.L2::128B.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float4 ld_global(const float4 *ptr)
{
    float4 ret;
    asm volatile("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_global(const float *ptr, const float &value)
{
    asm volatile("st.global.L1::no_allocate.f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

__device__ __forceinline__ void st_global(const float4 *ptr, const float4 &value)
{
    asm volatile("st.global.L1::no_allocate.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

__device__ __forceinline__ void serpentine_mma(
    float *C, float *A, float *B, int m_val, int n_val)
{

#pragma unroll
    for (int n = 0; n < n_val; n += 2)
    {
#pragma unroll
        for (int m = 0; m < m_val; m += 2)
        {
            int m_serp = (n % 4) ? (m_val - 2 - m) : m;

            float a0 = A[m_serp];
            float a1 = A[m_serp + 1];
            float b0 = B[n];
            float b1 = B[n + 1];

            C[(m_serp + 0) * n_val + (n + 0)] += a0 * b0;
            C[(m_serp + 1) * n_val + (n + 0)] += a1 * b0;
            C[(m_serp + 1) * n_val + (n + 1)] += a1 * b1;
            C[(m_serp + 0) * n_val + (n + 1)] += a0 * b1;
        }
    }
}

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void
sgemm_v4(const T *__restrict__ A, const T *__restrict__ B, float *__restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int m_val = 8;  // m_val per thread in M
    constexpr int n_val = 16; // n_val per thread in N
    constexpr int PAD = 4;
    constexpr int bM_pad = bM + PAD;
    constexpr int bN_pad = bN + PAD;

    // warp tile
    const int warp_m_idx = warp_id % 2;
    const int warp_n_idx = warp_id / 2;

    // thread tile
    const int lane_row = lane_id / 4; // 4 threads of a warp in N
    const int lane_col = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_m_idx * 64 + lane_row * m_val;
    const int col_offset1 = warp_n_idx * 32 + lane_col * n_val / 2;
    const int col_offset2 = col_offset1 + 64;

    // thread block swizzle
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int y = (oy << base) + (ox & ((1 << base) - 1));
    int x = (ox >> base);

    auto gA = A + x * bM * K;          // A is K-major
    auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = shared_memory + bM_pad * bK * NumPipe; // padding sA

    // Registers
    float reg_load_gA[8]; // 128 theads load 128*8 float, 1 thead load 8 float.
    float reg_load_gB[8];

    float reg_c[m_val * n_val] = {0.0f};
    float reg_a[m_val * 2];
    float reg_b[n_val * 2];

    int numK = (K + bK - 1) / bK;
    int k_tile_count = numK;
    int k_tile_next = 0;

    constexpr int nbN = bN / 4;

    // prefetch
    auto tile_gA = gA + k_tile_next * bK;
    auto tile_gB = gB;
    for (int k_pipe = 0; k_pipe < NumPipe - 1; ++k_pipe)
    {
        auto tile_sA = sA + k_pipe * bM_pad * bK;
        auto tile_sB = sB + k_pipe * bN_pad * bK;

        // load gA
#pragma unroll
        for (int i = 0; i < 8; ++i) // 128 / (128 / 8) = 8
        {
            int row = tid / 8 + i * 16;
            int col = tid % 8;
            reg_load_gA[i] = ld_global(tile_gA + row * K + col);
        }

        // load gB
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int global_row = k_tile_next * bK + i;
            reg_load_gB[i] = ld_global(tile_gB + global_row * N + tid);
        }

        // save to sA
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int row = tid / 8 + i * 16;
            int col = tid % 8;
            (tile_sA + col * bM_pad)[row] = reg_load_gA[i];
        }

        // save to sB
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            (tile_sB + i * bN_pad)[tid] = reg_load_gB[i];
        }

        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    int smem_pipe_read = 0;
    int smem_pipe_write = NumPipe - 1;

    auto tile_sA = sA + smem_pipe_read * bM_pad * bK;
    auto tile_sB = sB + smem_pipe_read * bN_pad * bK;

    __syncthreads();

    // prefetch smem to rmem
    reinterpret_cast<float4 *>(reg_a)[0] = reinterpret_cast<float4 *>(tile_sA + row_offset)[0];
    reinterpret_cast<float4 *>(reg_a)[1] = reinterpret_cast<float4 *>(tile_sA + row_offset)[1];

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        reg_b[i + 0] = (tile_sB + col_offset1)[i];
        reg_b[i + 8] = (tile_sB + col_offset2)[i];
    }

    // reinterpret_cast<float4 *>(reg_b)[0] = reinterpret_cast<float4 *>(tile_sB + col_offset1)[0];
    // reinterpret_cast<float4 *>(reg_b)[1] = reinterpret_cast<float4 *>(tile_sB + col_offset1)[1];
    // reinterpret_cast<float4 *>(reg_b)[2] = reinterpret_cast<float4 *>(tile_sB + col_offset2)[0];
    // reinterpret_cast<float4 *>(reg_b)[3] = reinterpret_cast<float4 *>(tile_sB + col_offset2)[1];

    while (k_tile_count > -(NumPipe - 1))
    {
#pragma unroll
        for (int tk = 0; tk < bK; ++tk)
        {
            if (tk == bK - 1)
            {
                // reg store to smem
                tile_sA = sA + smem_pipe_write * bM_pad * bK;
                tile_sB = sB + smem_pipe_write * bN_pad * bK;

                // save to sA
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    int row = tid / 8 + i * 16;
                    int col = tid % 8;
                    (tile_sA + col * bM_pad)[row] = reg_load_gA[i];
                }

                // save to sB
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    (tile_sB + i * bN_pad)[tid] = reg_load_gB[i];
                }
                __syncthreads();

                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == NumPipe - 1) ? 0 : smem_pipe_read + 1;
            }

            int reg_idx = tk % 2;
            int reg_next_idx = reg_idx ^ 1;
            int tk_next = (tk + 1) % bK;
            reinterpret_cast<float4 *>(reg_a)[reg_next_idx * 2 + 0] = reinterpret_cast<float4 *>(tile_sA + tk_next * bM_pad + row_offset)[0];
            reinterpret_cast<float4 *>(reg_a)[reg_next_idx * 2 + 1] = reinterpret_cast<float4 *>(tile_sA + tk_next * bM_pad + row_offset)[1];

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                reg_b[reg_next_idx * 16 + i + 0] = (tile_sB + tk_next * bN_pad + col_offset1)[i];
                reg_b[reg_next_idx * 16 + i + 8] = (tile_sB + tk_next * bN_pad + col_offset2)[i];
            }

            // reinterpret_cast<float4 *>(reg_b)[reg_next_idx * 4 + 0] = reinterpret_cast<float4 *>(tile_sB + tk_next * bN_pad + col_offset1)[0];
            // reinterpret_cast<float4 *>(reg_b)[reg_next_idx * 4 + 1] = reinterpret_cast<float4 *>(tile_sB + tk_next * bN_pad + col_offset1)[1];
            // reinterpret_cast<float4 *>(reg_b)[reg_next_idx * 4 + 2] = reinterpret_cast<float4 *>(tile_sB + tk_next * bN_pad + col_offset2)[0];
            // reinterpret_cast<float4 *>(reg_b)[reg_next_idx * 4 + 3] = reinterpret_cast<float4 *>(tile_sB + tk_next * bN_pad + col_offset2)[1];

            // auto frag_a = reg_a + reg_idx * m_val;
            // auto frag_b = reg_b + reg_idx * n_val;
            // serpentine_mma(reg_c, frag_a, frag_b, m_val, n_val);

#pragma unroll
            for (int i = 0; i < m_val; ++i)
            {
#pragma unroll
                for (int j = 0; j < n_val; ++j)
                {
                    reg_c[i * n_val + j] += reg_a[i + reg_idx * m_val] * reg_b[j + reg_idx * n_val];
                }
            }

            if (tk == 0)
            {
                // load from gmem to smem buffer
                auto tile_gA_write = gA + k_tile_next * bK;
                auto tile_gB_write = gB;
#pragma unroll
                for (int i = 0; i < 8; ++i) // 128 / (128 / 8) = 8
                {
                    int row = tid / 8 + i * 16;
                    int col = tid % 8;
                    reg_load_gA[i] = ld_global(tile_gA_write + row * K + col);
                }

#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    int global_row = k_tile_next * bK + i;
                    reg_load_gB[i] = ld_global(tile_gB_write + global_row * N + tid);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int local_row = row_offset + i;
#pragma unroll
        for (int j = 0; j < n_val / 4; ++j)
        {
            int local_col = col_offset1 + (j % 2) * 4 + 64 * (j / 2);
            // gC[local_row * N + local_col] = reg_c[i * n_val + j];
            reinterpret_cast<float4 *>(shared_memory + local_row * bN + local_col)[0] = reinterpret_cast<float4 *>(reg_c + i * n_val)[j]; // reuse shared memory
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = tid; i < bM * nbN; i += NumThreads)
    {
        int row = i / nbN;
        int col = i % nbN;
        reinterpret_cast<float4 *>(gC + row * N)[col] = reinterpret_cast<float4 *>(shared_memory + row * bN)[col];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    // M = M, K = N, N = K;
    // int temp = K;
    // K = N;
    // N = temp;
    // M = M;

    using T = float;
    constexpr int num_threads = 128;

    if (M == 8192 && K == 6144 && N == 4096)
    {
        // benchmark
        constexpr int blockM = 128;
        constexpr int blockN = 128;
        constexpr int blockK = 8;

        constexpr int numPipe = 2;
        int num_blockM = (M + blockM - 1) / blockM;
        int num_blockN = (N + blockN - 1) / blockN;

        // thread block swizzle
        constexpr int base = 3;
        num_blockM = num_blockM * (1 << base);
        num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

        dim3 block(num_threads);
        dim3 grid(num_blockM, num_blockN);
        int num_smem_values = max(blockM * blockN, (blockM + blockN) * blockK * numPipe);
        int smem_size = int(sizeof(T) * num_smem_values);
        auto kernel_fptr = sgemm_v4<blockM, blockN, blockK, num_threads, T, numPipe, base>;
        cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    }
    else
    {
        constexpr int blockM = 128;
        constexpr int blockN = 64;
        constexpr int blockK = 32;

        int num_blockM = (M + blockM - 1) / blockM;
        int num_blockN = (N + blockN - 1) / blockN;

        dim3 block(num_threads);
        dim3 grid(num_blockM, num_blockN);
        auto kernel_fptr = sgemm_v2<blockM, blockN, blockK, num_threads, T>;
        kernel_fptr<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}

int main()
{
    srand(1234);

    // leetGPU:  A is M×N, B is N×K, C is M×K
    // our impl: A is M×K, B is K×N, C is M×N

    int M = 4096, N = 4096, K = 4096;

    // M = 67, N = 67, K = 67;
    // M = 63, N = 31, K = 64;
    // M = 3, N = 3, K = 3;
    // M = 1027,N = 1027,K = 1027;
    M = 8192, N = 4096, K = 6144; // performance case

    using T = float;

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<float> h_C(M * N);
    thrust::host_vector<float> h_C1(M * N);
    thrust::host_vector<float> mma_res(M * N);
    thrust::host_vector<float> ref_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;
    thrust::device_vector<float> d_C1 = h_C1;

    solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    mma_res = d_C;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
    ref_res = d_C1;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 19.5e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}

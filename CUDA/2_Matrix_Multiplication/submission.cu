#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1>
__global__ void sgemm_base(const T *A, const T *B, float *C, int M, int N, int K)
{
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int m_tid = 16;
    constexpr int n_tid = NumThreads / m_tid;
    constexpr int num_tiledM = bM / m_tid;
    constexpr int num_tiledN = bN / n_tid;

    auto gA = A + x * bM * K;
    auto gC = C + x * bM * N + y * bN;

#pragma unroll
    for (int i = 0; i < num_tiledM; ++i)
    {
        int row_in_block = tid / n_tid * num_tiledM + i;
        int global_row = x * bM + row_in_block;
        if (global_row >= M)
            continue;
        auto tA = gA + row_in_block * K;

#pragma unroll
        for (int j = 0; j < num_tiledN; ++j)
        {
            int col_in_block = tid % n_tid * num_tiledN + j;
            int global_col = y * bN + col_in_block;
            if (global_col >= N)
                continue;
            auto tC = gC + row_in_block * N + col_in_block;
            float val_c = 0.0f;
            auto b_ptr = B + global_col;

#pragma unroll 8
            for (int k = 0; k < K; ++k)
            {
                val_c += static_cast<float>(tA[k] * b_ptr[k * N]);
            }
            tC[0] = val_c;
        }
    }
}

/**
 * Block Size: 64x64x64
 * Threads: 128 (4 Warps)
 *
 * A (M×K) is row-major, B (K×N) is row-major.
 *
 * 1. Block Tile: 64x64 (Computed by 1 CTA)
 * 2. Warp Tile:  32x32   (Computed by 1 Warp) -> Layout: 2x2 Warps
 * 3. Thread Tile:  4x8   (Computed by 1 Thread) -> Layout within Warp: 8x4 Threads
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v2(T *A, T *B, float *C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int m_val = 4; // m_val per thread in M
    const int n_val = 8; // n_val per thread in N

    // warp tile
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    // thread tile
    const int tid_row_in_warp = lane_id / 4;
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 32 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 32 + tid_col_in_warp * n_val;

    int x = blockIdx.x;
    int y = blockIdx.y;

    auto gA = A + x * bM * K;
    auto gB = B + y * bN * K;
    auto gC = C + x * bM * N + y * bN;

    __shared__ T sA[bM * bK];
    __shared__ T sB[bN * bK];

#pragma unroll
    for (int i = tid; i < bM * bK; i += NumThreads)
    {
        sA[i] = 0;
    }
#pragma unroll
    for (int i = tid; i < bN * bK; i += NumThreads)
    {
        sB[i] = 0;
    }
    __syncthreads();

    float reg_c[m_val * n_val] = {0.0f};
    float reg_a[m_val];
    float reg_b[n_val];

    int numK = K / bK;

    for (int k = 0; k < numK; ++k)
    {
        // copy A into sA, K-major, sA is 64 * 64
        auto sAgA = gA + k * bK;
        reinterpret_cast<float4 *>(sA + tid * bK)[0] = reinterpret_cast<float4 *>(sAgA + tid * K)[0];
        reinterpret_cast<float4 *>(sA + tid * bK)[1] = reinterpret_cast<float4 *>(sAgA + tid * K)[1];

        // copy B into sB, N-major
        auto sBgB = gB + k * bK;
        reinterpret_cast<float4 *>(sB + tid * bK)[0] = reinterpret_cast<float4 *>(sBgB + tid * K)[0];
        reinterpret_cast<float4 *>(sB + tid * bK)[1] = reinterpret_cast<float4 *>(sBgB + tid * K)[1];

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
                reg_b[j] = sB[(col_offset + j) * bK + tk];
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
        int global_row = row_offset + i;
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int global_col = col_offset + j;
            gC[global_row * N + global_col] = reg_c[i * n_val + j];
        }
    }
}

template <int N>
__device__ void cp_async_wait()
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
__device__ void cp_async_size16(T *smem, const T *gmem)
{
    __uint32_t smem_int_ptr = static_cast<__uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                 "l"(gmem),
                 "n"(sizeof(float4)));
}

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1>
__global__ void matrix_multiplication_kernel(const T *A, const T *B, float *C, int M, int N, int K)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int m_val = 8; // num values per threads in m
    constexpr int n_val = 4;
    // constexpr int m_tid = 8; // num threads per warp in m
    constexpr int n_tid = 4;

    auto gA = A + x * bM * K;
    auto gB = B + y * bN * K;
    auto gC = C + x * bM * N + y * bN;

    extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = shared_memory + bM * bK * NumPipe;

    int numK = K / bK;
    int k_tile_next = 0;

    constexpr int n_tid_col = bK / (sizeof(float4) / sizeof(T));
    int n_tid_row = NumThreads / n_tid_col;
    int m_step = bM / n_tid_row;
    int n_step = bN / n_tid_row;

    float mma_c[m_val * n_val] = {0.0f};
    float4 mma_a[m_val * n_tid_col];
    float4 mma_b[n_val * n_tid_col];

    // prefetch
    for (int k_pipe = 0; k_pipe < NumPipe - 1; ++k_pipe)
    {
        auto tile_gA = gA + k_tile_next * bK;
        auto tile_sA = sA + k_pipe * bM * bK;
#pragma unroll
        for (int i = 0; i < m_step; ++i)
        {
            int row = i * n_tid_row + tid / n_tid_col;
            int col = tid % n_tid_col;
            int swizzle_offset = row * n_tid_col + (col % 8) ^ (row / 8 % 8) + col / 8 * 8;
            auto smem_ptr = reinterpret_cast<float4 *>(tile_sA) + swizzle_offset;
            auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gA + row * K) + tid % n_tid_col;
            cp_async_size16(smem_ptr, gmem_ptr);
            // reinterpret_cast<float4 *>(tile_sA)[swizzle_offset] = reinterpret_cast<float4 *>(tile_gA + row * K)[tid % n_tid_col];
        }

        auto tile_gB = gB + k_tile_next * bK;
        auto tile_sB = sB + k_pipe * bN * bK;
#pragma unroll
        for (int i = 0; i < n_step; ++i)
        {
            int row = i * n_tid_row + tid / n_tid_col;
            int col = tid % n_tid_col;
            int swizzle_offset = row * n_tid_col + (col % 8) ^ (row / 4 % 4) + col / 8 * 8;
            auto smem_ptr = reinterpret_cast<float4 *>(tile_sB) + swizzle_offset;
            auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gB + row * K) + tid % n_tid_col;
            cp_async_size16(smem_ptr, gmem_ptr);
            // reinterpret_cast<float4 *>(tile_sB)[swizzle_offset] = reinterpret_cast<float4 *>(tile_gB + row * K)[tid % n_tid_col];
        }
        asm volatile("cp.async.commit_group;\n" ::);
        ++k_tile_next;
    }
    int smem_pipe_read = 0;
    int smem_pipe_write = NumPipe - 1;

    auto tile_sA = sA + smem_pipe_read * bM * bK;
    auto tile_sB = sB + smem_pipe_read * bN * bK;

    cp_async_wait<0>();
    __syncthreads();

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        auto tA = tile_sA + (lane_id / n_tid * m_val + i) * bK;
        mma_a[i + 0 * m_val] = reinterpret_cast<float4 *>(tA)[lane_id / n_tid];
    }
#pragma unroll
    for (int j = 0; j < n_val; ++j)
    {
        auto tB = tile_sB + (lane_id % n_tid * n_val + j + warp_id * n_tid * n_val) * bK;
        mma_b[j + 0 * n_val] = reinterpret_cast<float4 *>(tB)[lane_id % n_tid];
    }

    while (k_tile_next < numK)
    {
        // load from smem buffer to rmem and do gemm
#pragma unroll
        for (int tk = 0; tk < n_tid_col; ++tk) // float4
        {
            if (tk == n_tid_col - 1)
            {
                tile_sA = sA + smem_pipe_read * bM * bK;
                tile_sB = sB + smem_pipe_read * bN * bK;
                cp_async_wait<1>();
                __syncthreads();
            }

            int tk_next = (tk + 1) % n_tid_col;
#pragma unroll
            for (int i = 0; i < m_val; ++i)
            {
                auto tA = tile_sA + (lane_id / n_tid * m_val + i) * bK;
                mma_a[i + tk_next * m_val] = reinterpret_cast<float4 *>(tA)[((tk_next % 8) ^ (lane_id / n_tid)) + tk_next / 8 * 8];
            }
#pragma unroll
            for (int j = 0; j < n_val; ++j)
            {
                auto tB = tile_sB + (lane_id % n_tid * n_val + j + warp_id * n_tid * n_val) * bK;
                mma_b[j + tk_next * n_val] = reinterpret_cast<float4 *>(tB)[((tk_next % 8) ^ (lane_id % n_tid)) + tk_next / 8 * 8];
            }

            if (tk == 0)
            {
                // load from gmem to smem buffer
                auto tile_gA = gA + k_tile_next * bK;
                auto tile_sA = sA + smem_pipe_write * bM * bK;
#pragma unroll
                for (int i = 0; i < m_step; ++i)
                {
                    int row = i * n_tid_row + tid / n_tid_col;
                    int col = tid % n_tid_col;
                    int swizzle_offset = row * n_tid_col + (col % 8) ^ (row / 8 % 8) + col / 8 * 8;
                    auto smem_ptr = reinterpret_cast<float4 *>(tile_sA) + swizzle_offset;
                    auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gA + row * K) + tid % n_tid_col;
                    cp_async_size16(smem_ptr, gmem_ptr);
                }

                auto tile_gB = gB + k_tile_next * bK;
                auto tile_sB = sB + smem_pipe_write * bN * bK;
#pragma unroll
                for (int i = 0; i < n_step; ++i)
                {
                    int row = i * n_tid_row + tid / n_tid_col;
                    int col = tid % n_tid_col;
                    int swizzle_offset = row * n_tid_col + (col % 8) ^ (row / 4 % 4) + col / 8 * 8;
                    auto smem_ptr = reinterpret_cast<float4 *>(tile_sB) + swizzle_offset;
                    auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gB + row * K) + tid % n_tid_col;
                    cp_async_size16(smem_ptr, gmem_ptr);
                }
                asm volatile("cp.async.commit_group;\n" ::);

                ++k_tile_next;
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == NumPipe - 1) ? 0 : smem_pipe_read + 1;
            }

#pragma unroll
            for (int i = 0; i < m_val; ++i)
            {
#pragma unroll
                for (int j = 0; j < n_val; ++j)
                {
                    auto ra = mma_a[i + tk * m_val];
                    auto rb = mma_b[j + tk * n_val];
                    mma_c[i * n_val + j] += static_cast<float>(ra.x * rb.x);
                    mma_c[i * n_val + j] += static_cast<float>(ra.y * rb.y);
                    mma_c[i * n_val + j] += static_cast<float>(ra.z * rb.z);
                    mma_c[i * n_val + j] += static_cast<float>(ra.w * rb.w);
                }
            }
        }
    }

    // load from smem buffer to rmem and do gemm
#pragma unroll
    for (int tk = 0; tk < n_tid_col; ++tk) // float4
    {
        int tk_next = (tk + 1) % n_tid_col;
#pragma unroll
        for (int i = 0; i < m_val; ++i)
        {
            auto tA = tile_sA + (lane_id / n_tid * m_val + i) * bK;
            mma_a[i + tk_next * m_val] = reinterpret_cast<float4 *>(tA)[((tk_next % 8) ^ (lane_id / n_tid)) + tk_next / 8 * 8];
        }
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            auto tB = tile_sB + (lane_id % n_tid * n_val + j + warp_id * n_tid * n_val) * bK;
            mma_b[j + tk_next * n_val] = reinterpret_cast<float4 *>(tB)[((tk_next % 8) ^ (lane_id % n_tid)) + tk_next / 8 * 8];
        }

#pragma unroll
        for (int i = 0; i < m_val; ++i)
        {
#pragma unroll
            for (int j = 0; j < n_val; ++j)
            {
                auto ra = mma_a[i + tk * m_val];
                auto rb = mma_b[j + tk * n_val];
                mma_c[i * n_val + j] += static_cast<float>(ra.x * rb.x);
                mma_c[i * n_val + j] += static_cast<float>(ra.y * rb.y);
                mma_c[i * n_val + j] += static_cast<float>(ra.z * rb.z);
                mma_c[i * n_val + j] += static_cast<float>(ra.w * rb.w);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int index_m = lane_id / n_tid * m_val + i;
        int index_n = lane_id % n_tid * n_val + warp_id * n_tid * n_val;
        if (index_m >= M || index_n >= N)
            continue;
        auto tC = gC + index_m * N + index_n;
        reinterpret_cast<float4 *>(tC)[0] = reinterpret_cast<float4 *>(mma_c + i * n_val)[0];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    // M = M, K = N, N = K;
    int temp = K;
    K = N;
    N = temp;
    M = M;

    constexpr int blockM = 64;
    constexpr int blockN = 64;
    constexpr int blockK = 64;
    constexpr int numPipe = 2;
    using T = float;

    constexpr int num_threads = 128;
    int num_blockM = (M + blockM - 1) / blockM;
    int num_blockN = (N + blockN - 1) / blockN;

    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    if (M == 8192 && K == 6144 && N == 4096)
    {
        // benchmark, TODO: row_major B
        auto kernel_fptr = matrix_multiplication_kernel<blockM, blockN, blockK, num_threads, T, numPipe>;
        cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    }
    else
    {
        auto kernel_fptr = sgemm_base<blockM, blockN, blockK, num_threads, T, numPipe>;
        cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}

int main()
{
    srand(1234);

    int M = 4096;
    int N = 4096;
    int K = 4096;

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
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), N);
    ref_res = d_C1;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 66.9e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), N);
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

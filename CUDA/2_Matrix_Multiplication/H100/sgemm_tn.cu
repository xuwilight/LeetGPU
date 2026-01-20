#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v1(T *A, T *B, float *C, int M, int N, int K)
{
    const int tid = threadIdx.x;

    int x = blockIdx.x;
    int y = blockIdx.y;

    constexpr int m_tid = 16;         // 16 threads in M
    constexpr int n_tid = 8;          //  8 threads in N
    constexpr int m_val = bM / m_tid; //  8 per thread in M
    constexpr int n_val = bN / n_tid; // 16 per thread in N

    auto gA = A + x * bM * K;
    auto gB = B + y * bN * K;
    auto gC = C + x * bM * N + y * bN;

    int row_offset = tid / n_tid * m_val;
    int col_offset = tid % n_tid * n_val;

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int row_in_block = row_offset + i; // values contiguous
        if ((x * bM + row_in_block) >= M)
        {
            continue;
        }
        auto tA = gA + row_in_block * K;
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int col_in_block = col_offset + j;
            if ((y * bN + col_in_block) >= N)
            {
                continue;
            }
            auto tB = gB + col_in_block * K;
            auto tC = gC + row_in_block * N + col_in_block;

            float val_c = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                val_c += static_cast<float>(tA[k] * tB[k]);
            }
            tC[0] = val_c;
        }
    }
}

/**
 * Block Size: 128x128x8
 * Threads: 128 (4 Warps)
 *
 * 1. Block Tile: 128x128 (Computed by 1 CTA)
 * 2. Warp Tile:  64x64   (Computed by 1 Warp) -> Layout: 2x2 Warps
 * 3. Thread Tile: 8x16   (Computed by 1 Thread) -> Layout within Warp: 8x4 Threads
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v2(T *A, T *B, float *C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int m_val = 8;  // m_val per thread in M
    const int n_val = 16; // n_val per thread in N

    // warp tile
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    // thread tile
    const int tid_row_in_warp = lane_id / 4;
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 64 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 64 + tid_col_in_warp * n_val;

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
        // copy A into sA, K-major, sA is 128 * 8, 1 thread per line, 8 = 2 float4.
        auto sAgA = gA + k * bK;
        reinterpret_cast<float4 *>(sA + tid * bK)[0] = reinterpret_cast<float4 *>(sAgA + tid * K)[0];
        reinterpret_cast<float4 *>(sA + tid * bK)[1] = reinterpret_cast<float4 *>(sAgA + tid * K)[1];

        // copy B into sB, K-major
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


/**
 * bank conflicts not eliminated completely in 128*128*8
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v3(T *A, T *B, float *C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int m_val = 8;  // m_val per thread in M
    const int n_val = 16; // n_val per thread in N

    // warp tile
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    // thread tile
    const int tid_row_in_warp = lane_id / 4;
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 64 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 64 + tid_col_in_warp * n_val;

    int x = blockIdx.x;
    int y = blockIdx.y;

    auto gA = A + x * bM * K;
    auto gB = B + y * bN * K;
    auto gC = C + x * bM * N + y * bN;

    __shared__ T sA[bM * bK];
    __shared__ T sB[bN * bK];

    float reg_c[m_val * n_val] = {0.0f};
    float4 reg_a[m_val];
    float4 reg_b[n_val];

    int numK = K / bK;

    for (int k = 0; k < numK; ++k)
    {
        // copy A into sA, K-major, sA is 128 * 8, 1 thread per line, 8 = 2 float4.
        auto sAgA = gA + k * bK;
        int logical_row = tid / m_val;
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            int logical_col = tid % 4 * 2 + i; // 4 lines is a swizzle pattern
            int swizzle_col = (logical_row % 8) ^ logical_col;
            int new_row = swizzle_col / 2 + logical_row * m_val + (tid % m_val) / 4 * 4;
            int new_col = swizzle_col % 2;
            reinterpret_cast<float4 *>(sA + new_row * bK)[new_col] = reinterpret_cast<float4 *>(sAgA + tid * K)[i];
        }

        // copy B into sB, K-major
        auto sBgB = gB + k * bK;
        logical_row = tid / n_val;
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            int logical_col = tid % 4 * 2 + i; // 4 lines is a swizzle pattern
            int swizzle_col = (logical_row % 8) ^ logical_col;
            int new_row = swizzle_col / 2 + logical_row * n_val + (tid % n_val) / 4 * 4;
            int new_col = swizzle_col % 2;
            reinterpret_cast<float4 *>(sB + new_row * bK)[new_col] = reinterpret_cast<float4 *>(sBgB + tid * K)[i];
        }
        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < bK / 4; ++tk) // float4
        {
            // 1. Load A fragments into registers
            for (int i = 0; i < m_val; ++i)
            {
                int logical_row = (row_offset + i) / m_val;
                int logical_col = (row_offset + i) % 4 * 2 + tk;
                int swizzle_col = (logical_row % 8) ^ logical_col;
                int new_row = swizzle_col / 2 + logical_row * m_val + ((row_offset + i) % m_val) / 4 * 4;
                int new_col = swizzle_col % 2;
                reg_a[i] = reinterpret_cast<float4 *>(sA + new_row * bK)[new_col];
            }

            // 2. Load B fragments into registers
            for (int j = 0; j < n_val; ++j)
            {
                int logical_row = (col_offset + j) / n_val;
                int logical_col = (col_offset + j) % 4 * 2 + tk;
                int swizzle_col = (logical_row % 8) ^ logical_col;
                int new_row = swizzle_col / 2 + logical_row * n_val + ((col_offset + j) % n_val) / 4 * 4;
                int new_col = swizzle_col % 2;
                reg_b[j] = reinterpret_cast<float4 *>(sB + new_row * bK)[new_col];
            }

#pragma unroll
            for (int i = 0; i < m_val; ++i)
            {
#pragma unroll
                for (int j = 0; j < n_val; ++j)
                {
                    auto ra = reg_a[i];
                    auto rb = reg_b[j];
                    reg_c[i * n_val + j] += static_cast<float>(ra.x * rb.x);
                    reg_c[i * n_val + j] += static_cast<float>(ra.y * rb.y);
                    reg_c[i * n_val + j] += static_cast<float>(ra.z * rb.z);
                    reg_c[i * n_val + j] += static_cast<float>(ra.w * rb.w);
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

// nvcc sgemm_tn.cu -O3 -lcuda -lcublas -o sgemm_tn -arch=sm_90a && ./sgemm_tn
int main()
{
    srand(1234);

    int M = 4096;
    int N = 4096;
    int K = 5120;

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

    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 8;
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
    auto kernel_fptr = sgemm_v3<blockM, blockN, blockK, num_threads, T, numPipe, base>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel_fptr<<<grid, block, smem_size>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    mma_res = d_C;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), K, &beta, d_C1.data().get(), N);
    ref_res = d_C1;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 66.9e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), K, &beta, d_C1.data().get(), N);
        };

        std::function<void()> custom_func = [&]()
        {
            kernel_fptr<<<grid, block, smem_size>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}

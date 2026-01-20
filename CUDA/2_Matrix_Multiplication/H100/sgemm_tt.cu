#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1>
__global__ void sgemm_v1(const T *A, const T *B, float *C, int M, int N, int K)
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
            float val_c = 0.0f;
            auto b_ptr = gB + col_in_block;

            for (int k = 0; k < K; ++k)
            {
                val_c += static_cast<float>(tA[k] * b_ptr[k * N]);
            }
            tC[0] = val_c;
        }
    }
}

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

    const int m_val = 4; // m_val per thread in M
    const int n_val = 8; // n_val per thread in N

    // warp tile
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    // thread tile
    const int tid_row_in_warp = lane_id / 4; // 4 threads of a warp in N
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 32 + tid_row_in_warp * m_val;
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

/**
 * sgemm_v3 based on sgemm_v2.
 * 1. add swizzle to reduce bank conflicts.
 * 2. use float4.
 * 3. use thread block swizzle to improve L2 cache hit rate.
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v3(const T *A, const T *B, float *C, int M, int N, int K)
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
    const int tid_row_in_warp = lane_id / 4; // 4 threads of a warp in N
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 32 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 32 + tid_col_in_warp * n_val;

    // thread block swizzle
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int y = (oy << base) + (ox & ((1 << base) - 1));
    int x = (ox >> base);

    auto gA = A + x * bM * K;          // A is K-major
    auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    __shared__ T smem_buffer[bM * bK + bN * bK];
    T *sA = smem_buffer;
    T *sB = smem_buffer + bM * bK;

    float reg_c[m_val * n_val] = {0.0f};
    float reg_a[m_val];
    float reg_b[n_val];

    int numK = (K + bK - 1) / bK;

    for (int k = 0; k < numK; ++k)
    {
        // copy A into sA, K-major
        int n_tid_col = bK / 4; // use float4
#pragma unroll
        for (int i = tid; i < bM * n_tid_col; i += NumThreads)
        {
            int row = i / n_tid_col;
            int col = i % n_tid_col;

            int logical_row = (row / m_val) % 8;
            int swizzle_col = logical_row ^ col; // logical_row xor col
            reinterpret_cast<float4 *>(sA + row * bK)[swizzle_col] = reinterpret_cast<const float4 *>(gA + row * K + k * bK)[col];
        }

        // copy B into sB, N-major
        n_tid_col = bN / 4; // use float4
#pragma unroll
        for (int i = tid; i < n_tid_col * bK; i += NumThreads)
        {
            int row = i / n_tid_col;
            int col = i % n_tid_col;

            int global_row = k * bK + row;
            reinterpret_cast<float4 *>(sB + row * bN)[col] = reinterpret_cast<const float4 *>(gB + global_row * N)[col];
        }
        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < bK; ++tk)
        {
            // 1. Load A fragments into registers
            for (int i = 0; i < m_val; ++i)
            {
                int logical_row = ((row_offset + i) / m_val) % 8;
                int logical_col = logical_row ^ (tk / 4);
                reg_a[i] = sA[(row_offset + i) * bK + logical_col * 4 + tk % 4];
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
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int local_col = col_offset + j;
            smem_buffer[local_row * bN + local_col] = reg_c[i * n_val + j]; // reuse shared memory
        }
    }
    __syncthreads();

    int n_tid_col = bN / 4; // use float4
#pragma unroll
    for (int i = tid; i < bM * n_tid_col; i += NumThreads)
    {
        int row = i / n_tid_col;
        int col = i % n_tid_col;
        reinterpret_cast<float4 *>(gC + row * N)[col] = reinterpret_cast<float4 *>(smem_buffer + row * bN)[col];
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

/**
 * sgemm_v4 based on sgemm_v3.
 * 1. cp.async
 * 2. multi-stages
 */
template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1, int base = 0>
__global__ void sgemm_v4(const T *A, const T *B, float *C, int M, int N, int K)
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
    const int tid_row_in_warp = lane_id / 4; // 4 threads of a warp in N
    const int tid_col_in_warp = lane_id % 4;

    // The starting row (M) and col (N) for this thread's result tile
    const int row_offset = warp_row * 32 + tid_row_in_warp * m_val;
    const int col_offset = warp_col * 32 + tid_col_in_warp * n_val;

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
    T *sB = shared_memory + bM * bK * NumPipe;

    float reg_c[m_val * n_val] = {0.0f};
    float reg_a[m_val * bK];
    float reg_b[n_val * bK];

    int numK = (K + bK - 1) / bK;
    int k_tile_count = numK;
    int k_tile_next = 0;

    // prefetch
    for (int k_pipe = 0; k_pipe < NumPipe - 1; ++k_pipe)
    {
        auto tile_gA = gA + k_tile_next * bK;
        auto tile_sA = sA + k_pipe * bM * bK;

        // copy A into sA, K-major
        int n_tid_col = bK / 4; // use float4
#pragma unroll
        for (int i = tid; i < bM * n_tid_col; i += NumThreads)
        {
            int row = i / n_tid_col;
            int col = i % n_tid_col;

            int logical_row = (row / m_val) % 8;
            int swizzle_col = logical_row ^ col; // logical_row xor col
            auto smem_ptr = reinterpret_cast<float4 *>(tile_sA + row * bK) + swizzle_col;
            auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gA + row * K) + col;
            cp_async_size16(smem_ptr, gmem_ptr);
            // reinterpret_cast<float4 *>(sA + row * bK)[swizzle_col] = reinterpret_cast<const float4 *>(gA + row * K + k * bK)[col];
        }

        auto tile_gB = gB;
        auto tile_sB = sB + k_pipe * bN * bK;

        // copy B into sB, N-major
        n_tid_col = bN / 4; // use float4
#pragma unroll
        for (int i = tid; i < n_tid_col * bK; i += NumThreads)
        {
            int row = i / n_tid_col;
            int col = i % n_tid_col;

            int global_row = k_tile_next * bK + row;
            auto smem_ptr = reinterpret_cast<float4 *>(tile_sB + row * bN) + col;
            auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gB + global_row * N) + col;
            cp_async_size16(smem_ptr, gmem_ptr);
            // reinterpret_cast<float4 *>(sB + row * bN)[col] = reinterpret_cast<const float4 *>(gB + global_row * N)[col];
        }

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

    cp_async_wait<NumPipe - 2>(); // at least one stage is ready
    __syncthreads();

// prefetch smem to rmem
#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int logical_row = ((row_offset + i) / m_val) % 8;
        int logical_col = logical_row ^ (0 / 4);
        reg_a[i + 0 * m_val] = tile_sA[(row_offset + i) * bK + logical_col * 4 + 0 % 4];
    }
#pragma unroll
    for (int j = 0; j < n_val; ++j)
    {
        reg_b[j + 0 * n_val] = tile_sB[0 * bN + col_offset + j];
    }

    while (k_tile_count > -(NumPipe - 1))
    {
#pragma unroll
        for (int tk = 0; tk < bK; ++tk)
        {
            if (tk == bK - 1)
            {
                tile_sA = sA + smem_pipe_read * bM * bK;
                tile_sB = sB + smem_pipe_read * bN * bK;
                cp_async_wait<NumPipe - 2>();
                __syncthreads();
            }

            int tk_next = (tk + 1) % bK;
#pragma unroll
            for (int i = 0; i < m_val; ++i)
            {
                int logical_row = ((row_offset + i) / m_val) % 8;
                int logical_col = logical_row ^ (tk_next / 4);
                reg_a[i + tk_next * m_val] = tile_sA[(row_offset + i) * bK + logical_col * 4 + tk_next % 4];
            }
#pragma unroll
            for (int j = 0; j < n_val; ++j)
            {
                reg_b[j + tk_next * n_val] = tile_sB[tk_next * bN + col_offset + j];
            }

            if (tk == 0)
            {
                // load from gmem to smem buffer
                auto tile_gA = gA + k_tile_next * bK;
                auto tile_sA = sA + smem_pipe_write * bM * bK;
                int n_tid_col = bK / 4; // use float4
#pragma unroll
                for (int i = tid; i < bM * n_tid_col; i += NumThreads)
                {
                    int row = i / n_tid_col;
                    int col = i % n_tid_col;

                    int logical_row = (row / m_val) % 8;
                    int swizzle_col = logical_row ^ col; // logical_row xor col
                    auto smem_ptr = reinterpret_cast<float4 *>(tile_sA + row * bK) + swizzle_col;
                    auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gA + row * K) + col;
                    cp_async_size16(smem_ptr, gmem_ptr);
                }

                auto tile_gB = gB;
                auto tile_sB = sB + smem_pipe_write * bN * bK;
                n_tid_col = bN / 4; // use float4
#pragma unroll
                for (int i = tid; i < n_tid_col * bK; i += NumThreads)
                {
                    int row = i / n_tid_col;
                    int col = i % n_tid_col;

                    int global_row = k_tile_next * bK + row;
                    auto smem_ptr = reinterpret_cast<float4 *>(tile_sB + row * bN) + col;
                    auto gmem_ptr = reinterpret_cast<const float4 *>(tile_gB + global_row * N) + col;
                    cp_async_size16(smem_ptr, gmem_ptr);
                }
                asm volatile("cp.async.commit_group;\n" ::);

                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == NumPipe - 1) ? 0 : smem_pipe_read + 1;
            }

            for (int i = 0; i < m_val; ++i)
            {
                for (int j = 0; j < n_val; ++j)
                {
                    reg_c[i * n_val + j] += reg_a[i + tk * m_val] * reg_b[j + tk * n_val];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < m_val; ++i)
    {
        int local_row = row_offset + i;
#pragma unroll
        for (int j = 0; j < n_val; ++j)
        {
            int local_col = col_offset + j;
            shared_memory[local_row * bN + local_col] = reg_c[i * n_val + j]; // reuse shared memory
        }
    }
    __syncthreads();

    int n_tid_col = bN / 4; // use float4
#pragma unroll
    for (int i = tid; i < bM * n_tid_col; i += NumThreads)
    {
        int row = i / n_tid_col;
        int col = i % n_tid_col;
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

    constexpr int blockM = 64;
    constexpr int blockN = 64;
    constexpr int blockK = 32;
    using T = float;

    constexpr int num_threads = 128;

    if (M == 8192 && K == 6144 && N == 4096)
    {
        // benchmark
        constexpr int numPipe = 3;
        int num_blockM = (M + blockM - 1) / blockM;
        int num_blockN = (N + blockN - 1) / blockN;

        // thread block swizzle
        constexpr int base = 3;
        num_blockM = num_blockM * (1 << base);
        num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

        dim3 block(num_threads);
        dim3 grid(num_blockM, num_blockN);
        int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
        auto kernel_fptr = sgemm_v4<blockM, blockN, blockK, num_threads, T, numPipe, base>;
        cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    }
    else
    {
        int num_blockM = (M + blockM - 1) / blockM;
        int num_blockN = (N + blockN - 1) / blockN;

        dim3 block(num_threads);
        dim3 grid(num_blockM, num_blockN);
        auto kernel_fptr = sgemm_v2<blockM, blockN, blockK, num_threads, T>;
        kernel_fptr<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}

// nvcc sgemm_tt.cu -O3 -lcuda -lcublas -o sgemm_tt -arch=sm_90a && ./sgemm_tt
// cublas time = 8.150183 ms, TFLPOS = 50.589891, mfu = 0.756202
// mma time = 11.728918 ms, TFLPOS = 35.153871, mfu = 0.525469
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
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B.data().get(), N, d_A.data().get(), K, &beta, d_C1.data().get(), N);
    ref_res = d_C1;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 66.9e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B.data().get(), N, d_A.data().get(), K, &beta, d_C1.data().get(), N);
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

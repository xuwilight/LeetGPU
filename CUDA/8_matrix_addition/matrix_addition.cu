#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

__global__ void matrix_add_base(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

template <int bM, int bN>
__global__ void matrix_add(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    int tid = threadIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;

    auto gA = A + x * bM * N + y * bN;
    auto gB = B + x * bM * N + y * bN;
    auto gC = C + x * bM * N + y * bN;

    auto gA4 = reinterpret_cast<const float4 *>(gA);
    auto gB4 = reinterpret_cast<const float4 *>(gB);
    auto gC4 = reinterpret_cast<float4 *>(gC);

    int N4 = N >> 2;

    int col = tid % 32;
    int row_base = tid / 32;

#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        int row = row_base + i * 4;
        int offset = row * N4 + col;
        auto a = __ldg(gA4 + offset);
        auto b = __ldg(gB4 + offset);
        float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        gC4[offset] = c;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    if (N == 4096)
    {
        constexpr int num_thread = 128;
        constexpr int blockM = 128;
        constexpr int blockN = 128;
        int num_M = (N + blockM - 1) / blockM;
        int num_N = (N + blockN - 1) / blockN;

        dim3 block(num_thread);
        dim3 grid(num_M, num_N);
        matrix_add<blockM, blockN><<<grid, block>>>(A, B, C, N);
    }
    else
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
        matrix_add_base<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    }
}

void validate(float *a, float *b, float *c, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        float t = a[i] + b[i];
        if (abs(t - c[i]) > 1e-12)
        {
            printf("index = %d, expect = %f, real = %f\n", i, t, c[i]);
            return;
        }
    }
    printf("calculate correct\n");
}

// nvcc -O3 -arch=sm_80 matrix_addition.cu -o matrix_addition && ./matrix_addition
int main()
{
    float u = 1000.0;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-u, u);

    int M = 4096, N = 4096;

    using T = float;

    thrust::host_vector<T> h_A(M * N);
    thrust::host_vector<T> h_B(M * N);
    thrust::host_vector<float> h_C(M * N);

    for (int i = 0; i < M * N; ++i)
    {
        h_A[i] = static_cast<T>(dis(gen));
        h_B[i] = static_cast<T>(dis(gen));
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;

    solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), N);

    h_C = d_C;
    validate(h_A.data(), h_B.data(), h_C.data(), N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int count = 10;
    float elapsed_time;

    for (int i = 0; i < 10; ++i)
        solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    for (int i = 0; i < count; ++i)
    {
        solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("benchmark = %.9f ms\n", elapsed_time / count);

    return 0;
}

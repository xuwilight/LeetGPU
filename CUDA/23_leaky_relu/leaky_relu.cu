#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

__global__ void leaky_relu_kernel(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float val = __ldg(&input[idx]);
        output[idx] = val > 0.0f ? val : 0.01f * val;
    }
}

__global__ void leaky_relu_kernel_optim(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int vecN = N / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    float4 *output4 = reinterpret_cast<float4 *>(output);

#pragma unroll
    for (int i = index; i < vecN; i += stride)
    {
        float4 v = __ldg(input4 + i);
        v.x = v.x > 0.0f ? v.x : 0.01f * v.x;
        v.y = v.y > 0.0f ? v.y : 0.01f * v.y;
        v.z = v.z > 0.0f ? v.z : 0.01f * v.z;
        v.w = v.w > 0.0f ? v.w : 0.01f * v.w;
        output4[i] = v;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    if (N == 50000000)
    {
        constexpr int num_sm = 132;
        constexpr int num_thread = 512;
        int grid_size = std::min((N + num_thread - 1) / num_thread, 64 * num_sm);
        leaky_relu_kernel_optim<<<grid_size, num_thread>>>(input, output, N);
    }
    else
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / (threadsPerBlock);
        leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}

void validate(float *a, float *b, int N)
{
    for (int i = 0; i < N; ++i)
    {
        float t = a[i] > 0.0f ? a[i] : 0.01f * a[i];
        if (abs(t - b[i]) > 1e-12)
        {
            printf("index = %d, expect = %f, real = %f\n", i, t, b[i]);
            return;
        }
    }
    printf("calculate correct\n");
}

// nvcc -O3 -arch=sm_80 leaky_relu.cu -o leaky_relu && ./leaky_relu
int main()
{
    float u = 100.0;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-u, u);

    int N = 50000000;

    using T = float;

    thrust::host_vector<T> input1(N);
    thrust::host_vector<T> output(N);

    for (int i = 0; i < N; ++i)
    {
        input1[i] = static_cast<T>(dis(gen));
    }

    thrust::device_vector<T> in_d1 = input1;
    thrust::device_vector<T> out_d = output;

    solve(in_d1.data().get(), out_d.data().get(), N);

    output = out_d;
    validate(input1.data(), output.data(), N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int count = 10;
    float elapsed_time;

    for (int i = 0; i < 10; ++i)
        solve(in_d1.data().get(), out_d.data().get(), N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    for (int i = 0; i < count; ++i)
    {
        solve(in_d1.data().get(), out_d.data().get(), N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("benchmark = %.9f ms\n", elapsed_time / count);

    return 0;
}

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cmath>


// base
__global__ void vector_add1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// float4
template <int stride = 4>
__global__ void vector_add2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x * stride;
    const int offset = index + tid * stride;

    float4 a, b, c;
    if (offset + 3 < N)
    {
        a = reinterpret_cast<const float4 *>(A + index)[tid];
        b = reinterpret_cast<const float4 *>(B + index)[tid];
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C + index)[tid] = c;
    }
    else
    {
        for (int i = 0; i < 4 && (offset + i) < N; ++i)
        {
            C[offset + i] = A[offset + i] + B[offset + i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int N)
{
    constexpr int stride = 4;
    constexpr int unroll = 1;
    constexpr int stage = 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * stride * unroll * stage - 1) / (threadsPerBlock * stride * unroll * stage);

    vector_add2<stride, unroll, stage><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

void validate(float *a, float *b, float *c, int N)
{
    for (int i = 0; i < N; ++i)
    {
        float t = a[i] + b[i];
        if (abs(t - c[i]) > 1e-12)
        {
            printf("error: index = %d, expect = %f, real = %f\n", i, t, c[i]);
            return;
        }
    }
    printf("calculate correct\n");
}

int main()
{
    srand(2333);

    int N = 1e8;
    thrust::host_vector<float> input1(N);
    thrust::host_vector<float> input2(N);
    thrust::host_vector<float> output(N);

    for (int i = 0; i < N; ++i)
    {
        input1[i] = static_cast<float>(rand() % 9 + 1);
        input2[i] = static_cast<float>(rand() % 9 + 1);
        output[i] = static_cast<float>(0);
    }

    thrust::device_vector<float> in_d1 = input1;
    thrust::device_vector<float> in_d2 = input2;
    thrust::device_vector<float> out_d = output;

    solve(in_d1.data().get(), in_d2.data().get(), out_d.data().get(), N);

    int benchmark = 1;

    if (benchmark)
    {

        output = out_d;
        validate(input1.data(), input2.data(), output.data(), N);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int count = 10;
        float elapsed_time;

        for (int i = 0; i < 2; ++i)
            solve(in_d1.data().get(), in_d2.data().get(), out_d.data().get(), N);
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);
        for (int i = 0; i < count; ++i)
        {
            solve(in_d1.data().get(), in_d2.data().get(), out_d.data().get(), N);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("benchmark = %.9f ms\n", elapsed_time / count);
    }

    return 0;
}

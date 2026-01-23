#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cublas_v2.h>
#include <random>

__global__ void ref_sum(const float *__restrict__ input, float *__restrict__ output, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        sum += __ldg(input + i);
    }
    if (threadIdx.x == 0)
    {
        output[0] = (float)sum;
    }
}

template <int num_thread>
__global__ void reduce_kernel(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warp = num_thread / 32;

    __shared__ float smem[32];

    float sum = 0.0f;

#pragma unroll
    for (int i = index; i < N; i += stride)
    {
        sum += __ldg(input + i);
    }

    // warp reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // block reduction
    if (lane_id == 0)
    {
        smem[warp_id] = sum; // warp_id need < 32
    }
    __syncthreads();

    sum = (threadIdx.x < num_warp) ? smem[lane_id] : 0.0f;

    if (warp_id == 0)
    {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(output, sum);
    }
}

template <int num_thread>
__global__ void reduce_kernel_float4(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warp = num_thread / 32;

    __shared__ float smem[32];

    float sum = 0.0f;

    int vecN = N / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);

#pragma unroll
    for (int i = index; i < vecN; i += stride)
    {
        float4 v = __ldg(input4 + i);
        sum += v.x + v.y + v.z + v.w;
    }

    // int tailStart = vecN * 4;
    // for (int i = tailStart + index; i < N; i += stride) {
    //     sum += input[i];
    // }

    // warp reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // block reduction
    if (lane_id == 0)
    {
        smem[warp_id] = sum; // warp_id need < 32
    }
    __syncthreads();

    sum = (threadIdx.x < num_warp) ? smem[lane_id] : 0.0f;

    if (warp_id == 0)
    {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(output, sum);
    }
}

void solve(const float *input, float *output, int N)
{
    if (N == 1024 * 4096)
    {
        constexpr int num_sm = 132;
        constexpr int num_thread = 256;
        int grid_size = std::min((N + num_thread - 1) / num_thread, 8 * num_sm);
        reduce_kernel_float4<num_thread><<<grid_size, num_thread>>>(input, output, N);
    }
    else
    {
        ref_sum<<<1, 32>>>(input, output, N);
    }
}

// nvcc -O3 -arch=sm_80 reduction.cu -o reduction && ./reduction
int main()
{
    float u = 1000.0;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0, u);

    int N = 1024 * 4096;
    thrust::host_vector<float> input1(N);
    thrust::host_vector<float> output(1);
    thrust::host_vector<float> output_ref(1);

    for (int i = 0; i < N; ++i)
    {
        input1[i] = static_cast<float>(dis(gen));
    }

    thrust::device_vector<float> in_d1 = input1;
    thrust::device_vector<float> out_d = output;
    thrust::device_vector<float> out_ref = output_ref;

    solve(in_d1.data().get(), out_d.data().get(), N);

    int benchmark = 1;

    if (benchmark)
    {
        ref_sum<<<1, 32>>>(in_d1.data().get(), out_ref.data().get(), N);
        output = out_d;
        output_ref = out_ref;
        if (abs(output_ref[0] - output[0]) < 1e-5 + 1e-5 * abs(output_ref[0]))
        {
            printf("calculate correct\n");
        }
        else
        {
            printf("calculate error, expected = %f, got = %f\n", output_ref[0], output[0]);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int count = 10;
        float elapsed_time;

        for (int i = 0; i < 2; ++i)
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
    }

    return 0;
}

#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

__device__ __forceinline__ float4 add_float4(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void vector_add1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

template <int stride = 4, int unroll = 4>
__global__ void vector_add2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x * stride;
    const int offset = index + tid * stride;

    float4 a, b, c;
    if (offset + 3 < N)
    {
        a = __ldg(reinterpret_cast<const float4 *>(A + index) + tid);
        b = __ldg(reinterpret_cast<const float4 *>(B + index) + tid);
        c = add_float4(a, b);
        reinterpret_cast<float4 *>(C + index)[tid] = c;
    }
    else
    {
        for (int i = 0; i < 4 && (offset + i) < N; ++i)
        {
            C[offset + i] = __ldg(A + offset + i) + __ldg(B + offset + i);
        }
    }
}

template <int stride = 4, int unroll = 4>
__global__ void vector_add3(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x * unroll * stride;
    const int base_idx = block_offset + tid * stride;

#pragma unroll
    for (int u = 0; u < unroll; ++u)
    {
        int offset = base_idx + u * blockDim.x * stride;
        if (offset + 3 < N)
        {
            float4 a = __ldg(reinterpret_cast<const float4 *>(A + offset));
            float4 b = __ldg(reinterpret_cast<const float4 *>(B + offset));
            float4 c = add_float4(a, b);
            *reinterpret_cast<float4 *>(C + offset) = c;
        }
        else
        {
            for (int i = 0; i < 4 && (offset + i) < N; ++i)
            {
                C[offset + i] = A[offset + i] + B[offset + i];
            }
        }
    }
}


template <int STRIDE, int UNROLL>
__global__ void vector_add4(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N)
{
    const float4 *A4 = reinterpret_cast<const float4 *>(A);
    const float4 *B4 = reinterpret_cast<const float4 *>(B);
    float4 *C4 = reinterpret_cast<float4 *>(C);

    size_t vec_n = N / 4;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 a_reg[UNROLL];
    float4 b_reg[UNROLL];
    float4 c_reg[UNROLL];

    size_t i = idx;
    for (; i + STRIDE * (UNROLL - 1) < vec_n; i += STRIDE * UNROLL)
    {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
        {
            a_reg[u] = __ldg(A4 + i + u * STRIDE);
            b_reg[u] = __ldg(B4 + i + u * STRIDE);
            c_reg[u] = add_float4(a_reg[u], b_reg[u]);
            C4[i + u * STRIDE] = c_reg[u];
        }
    }

    for (; i < vec_n; i += STRIDE)
    {
        float4 a = __ldg(A4 + i);
        float4 b = __ldg(B4 + i);
        C4[i] = add_float4(a, b);
    }

    if (blockIdx.x == 0)
    {
        size_t remainder_start = vec_n * 4;
        for (size_t k = remainder_start + threadIdx.x; k < N; k += blockDim.x)
        {
            C[k] = A[k] + B[k];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    constexpr int stride = 4;
    constexpr int unroll = 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * stride * unroll- 1) / (threadsPerBlock * stride * unroll);
    vector_add3<stride, unroll><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaDeviceSynchronize();
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve1(const float *A, const float *B, float *C, int N)
{
    constexpr int threadsPerBlock = 256;
    constexpr int blocksPerGrid = 132 * 8; // 132 SMs * 8 Blocks
    constexpr int UNROLL = 4;
    constexpr int STRIDE = blocksPerGrid * threadsPerBlock;
    vector_add3<STRIDE, UNROLL><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

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

// nvcc -O3 vector_add.cu -o vector_add
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
    // solve1(in_d1.data().get(), in_d2.data().get(), out_d.data().get(), N);

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

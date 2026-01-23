#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void validate(float *a, float *b, float *c, int N)
{
    for (int i = 0; i < N; ++i)
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

__global__ void invert_kernel0(unsigned char *image, int width, int height)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 4;

    if (index >= width * height * 4)
        return;
    char4 pix = reinterpret_cast<char4 *>(image + index)[tid];

    pix.x = 255 - pix.x;
    pix.y = 255 - pix.y;
    pix.z = 255 - pix.z;

    reinterpret_cast<char4 *>(image + index)[tid] = pix;
}

__global__ void invert_kernel(unsigned char *image, int width, int height)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 4 * 4;

    if (index >= width * height * 4)
        return;

    unsigned int mask = 0x00FFFFFF;
    uint4 pix = __ldg(reinterpret_cast<uint4 *>(image + index) + tid);

    pix.x = (~pix.x & mask) | (pix.x & 0xFF000000);
    pix.y = (~pix.y & mask) | (pix.y & 0xFF000000);
    pix.z = (~pix.z & mask) | (pix.z & 0xFF000000);
    pix.w = (~pix.w & mask) | (pix.w & 0xFF000000);

    reinterpret_cast<uint4 *>(image + index)[tid] = pix;
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char *image, int width, int height)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
}

// nvcc -O3 -arch=sm_80 color_inversion.cu -o color_inversion && ./color_inversion
int main()
{
    srand(2333);

    int w = 4096;
    int h = 5120;
    int N = w * h * 4;

    using T = unsigned char;

    thrust::host_vector<T> input1(N);
    thrust::host_vector<T> output(N);

    for (int i = 0; i < N; ++i)
    {
        input1[i] = static_cast<T>(rand() % 9 + 1);
        // printf("%d ", input1[i]);
    }

    // printf("\n");

    thrust::device_vector<T> in_d1 = input1;
    thrust::device_vector<T> out_d = output;

    solve(in_d1.data().get(), w, h);

    output = in_d1;

    for (int i = 0; i < w * h * 4; ++i)
    {
        if ((i + 1) % 4 == 0)
        {
            if (input1[i] != output[i])
            {
                printf("error\n");
                break;
            }
        }
        else
        {
            if (input1[i] + output[i] != 255)
            {
                printf("error\n");
                break;
            }
        }
    }
    printf("correct\n");

    // validate(input1.data(), input2.data(), output.data(), N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int count = 10;
    float elapsed_time;

    for (int i = 0; i < 10; ++i)
        solve(in_d1.data().get(), w, h);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    for (int i = 0; i < count; ++i)
    {
        solve(in_d1.data().get(), w, h);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("benchmark = %.9f ms\n", elapsed_time / count);

    return 0;
}

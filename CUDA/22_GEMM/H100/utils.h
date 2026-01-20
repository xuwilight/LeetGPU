#include <iostream>
#include <functional>
#include <cuda.h>

void run_benchmark(std::function<void()> kernel_func,
                   const char *kernel_name,
                   float flops, float MAX_FLOPS, int num_runs = 100)
{
    // warm up
    kernel_func();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time_ms = 0;
    for (int i = 0; i < num_runs; ++i)
    {
        cudaEventRecord(start);
        kernel_func();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_ms += elapsed_ms;
    }
    float avg_time_ms = total_time_ms / num_runs;
    float TFPOS = flops / avg_time_ms * 1000;
    float MFU = TFPOS / MAX_FLOPS;
    printf("%s time = %f ms, TFLPOS = %f, mfu = %f\n", kernel_name, avg_time_ms, TFPOS / 1e12, MFU);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T, bool row_major_c = true>
void test_gemm(T *ref_result, T *mma_result, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float mma_v = 0.0f;
            float ref_v = static_cast<float>(ref_result[i * N + j]);
            if (row_major_c)
            {
                mma_v = static_cast<float>(mma_result[i * N + j]);
            }
            else
            {
                mma_v = static_cast<float>(mma_result[j * M + i]);
            }

            if (std::isnan(mma_v) || abs(ref_v - mma_v) > 1e-5)
            {
                printf("i = %d, j = %d, ref = %f, mma = %f \n", i, j, ref_v, mma_v);
                return;
            }
            // printf("i = %d, j = %d, ref = %f, mma = %f \n", i, j, ref_v, mma_v);
        }
    }
    printf("gemm success\n");
}

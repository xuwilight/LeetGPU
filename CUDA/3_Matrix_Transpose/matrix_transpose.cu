#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cublas_v2.h>
#include <random>

template <int TILE_SIZE, int NUM_THREAD>
__global__ void transpose_optimized(const float *__restrict__ idata, float *__restrict__ odata, int rows, int cols)
{
    __shared__ float smem[TILE_SIZE][TILE_SIZE + 1];

    int tid = threadIdx.x;

    int tid_row = tid / 32;
    int tid_col = tid % 32;

    constexpr int row_step = NUM_THREAD / TILE_SIZE;
    constexpr int num_rows = TILE_SIZE / row_step;

    int row_base = blockIdx.x * TILE_SIZE;
    int col_base = blockIdx.y * TILE_SIZE;

#pragma unroll
    for (int i = 0; i < num_rows; ++i)
    {
        int row = row_base + i * row_step + tid_row;
        int col = col_base + tid_col;
        if (row < rows && col < cols)
        {
            smem[i * row_step + tid_row][tid_col] = __ldg(idata + row * cols + col);
        }
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < num_rows; ++i)
    {
        int col = col_base + i * row_step + tid_row;
        int row = row_base + tid_col;
        if (row < rows && col < cols)
        {
            odata[col * rows + row] = smem[tid_col][i * row_step + tid_row];
        }
    }
}

__global__ void matrix_transpose_kernel(const float *input, float *output, int rows, int cols)
{
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);

    if (row >= rows || col >= cols)
        return;

    float val = input[row * cols + col];
    output[col * rows + row] = val;
}

extern "C" void solve(const float *input, float *output, int rows, int cols)
{
    if (rows == 7000 && cols == 6000)
    {
        constexpr int tile_size = 32;
        constexpr int num_thread = 128;
        dim3 block_dim(num_thread);
        dim3 grid_dim((rows + tile_size - 1) / tile_size, (cols + tile_size - 1) / tile_size);
        transpose_optimized<tile_size, num_thread><<<grid_dim, block_dim>>>(input, output, rows, cols);
    }
    else
    {
        int tile_size = 16;
        dim3 block_dim(tile_size, tile_size);
        dim3 grid_dim((rows + tile_size - 1) / tile_size, (cols + tile_size - 1) / tile_size);
        matrix_transpose_kernel<<<grid_dim, block_dim>>>(input, output, rows, cols);
    }
}

void validate(float *src, float *dst, int row, int col)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            auto f1 = src[i * col + j];
            auto f2 = dst[j * row + i];
            if (abs(f1 - f2) > 1e-12)
            {
                printf("error: i = %d, j = %d, expect = %f, real = %f\n", i, j, f1, f2);
                return;
            }
        }
    }
    printf("correct\n");
}

// nvcc -O3 -arch=sm_80 matrix_transpose.cu -o matrix_transpose && ./matrix_transpose
int main()
{
    float u = 10.0;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-u, u);

    int rows = 7000;
    int cols = 6000;

    int N = rows * cols;
    thrust::host_vector<float> input1(N);
    thrust::host_vector<float> output(N);

    for (int i = 0; i < N; ++i)
    {
        input1[i] = static_cast<float>(dis(gen));
        output[i] = static_cast<float>(0);
    }

    thrust::device_vector<float> in_d1 = input1;
    thrust::device_vector<float> out_d = output;

    solve(in_d1.data().get(), out_d.data().get(), rows, cols);

    int benchmark = 1;

    if (benchmark)
    {
        output = out_d;
        validate(input1.data(), output.data(), rows, cols);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int count = 10;
        float elapsed_time;

        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);
        for (int i = 0; i < count; ++i)
        {
            solve(in_d1.data().get(), out_d.data().get(), rows, cols);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("benchmark = %.9f ms\n", elapsed_time / count);
    }

    return 0;
}

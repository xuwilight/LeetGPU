#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include "utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

// (A * B)^T = B^T * A^T
template <class TA, class TB, class TC>
void cutlass_sgemm_tt(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{

    using cutlass_simt_sgemm_256x128_8x4_tt_align1_base =
        typename cutlass::gemm::kernel::DefaultGemmUniversal<
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed B operand
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed A operand
            float, cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,

            cutlass::epilogue::thread::LinearCombination<
                float,
                1,
                float,
                float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            4,
            cutlass::arch::OpMultiplyAdd>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_sgemm_256x128_8x4_tt_align1_base>;

    Gemm gemm;

    long long int batch_stride_A = static_cast<long long int>(M) * K;
    long long int batch_stride_B = static_cast<long long int>(K) * N;
    long long int batch_stride_C = static_cast<long long int>(M) * N;
    long long int batch_stride_D = static_cast<long long int>(M) * N;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,                        // 模式：标准 GEMM
        {M, N, K},                                                      // Problem Size
        1,                                                              // Batch Count (设为1表示非Batched)
        {alpha, beta},                                                  // Epilogue 参数
        A, B, C, C,                                                     // 指针
        batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, // Batch Strides
        lda, ldb, ldc, ldc                                              // Leading Dimension Strides
    );

    gemm(args);
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    cutlass_sgemm_tt(M, N, K, 1.0f, A, K, B, N, 0.0f, C, M);
    cudaDeviceSynchronize();
}

// nvcc cutlass_sgemm.cu -o cutlass -lcuda -lcublas -arch=sm_80 -O3 -I ../../cutlass-4.1/include  --ptxas-options=-v --expt-relaxed-constexpr  && ./cutlass
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
        // printf(" %f ", h_A[i]);
    }
    // printf("\n");
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
        // printf(" %f ", h_B[i]);
    }
    // printf("\n");

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;
    thrust::device_vector<float> d_C1 = h_C1;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
    ref_res = d_C1;

    solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    mma_res = d_C;

    test_gemm<T, false>(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 0;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 66.9e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
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

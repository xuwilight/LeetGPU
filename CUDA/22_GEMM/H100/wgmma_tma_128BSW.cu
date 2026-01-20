#include <iostream>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils.h"

__device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ int canonical_warp_idx_sync()
{
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__device__ void fence_view_async_shared()
{
    asm volatile(
        "{\n\t"
        "fence.proxy.async.shared::cta; \n"
        "}" ::);
}

__device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

// GMMA 64x128x16 F16+=F16*F16
template <int scale_D = 1, int scaleA = 1, int scaleB = 1, int tnspA = 0, int tnspB = 1>
__device__ static void
fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t *c)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " p,   %35, %36, %37, %38;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]),
          "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7]),
          "+r"(c[8]), "+r"(c[9]), "+r"(c[10]), "+r"(c[11]),
          "+r"(c[12]), "+r"(c[13]), "+r"(c[14]), "+r"(c[15]),
          "+r"(c[16]), "+r"(c[17]), "+r"(c[18]), "+r"(c[19]),
          "+r"(c[20]), "+r"(c[21]), "+r"(c[22]), "+r"(c[23]),
          "+r"(c[24]), "+r"(c[25]), "+r"(c[26]), "+r"(c[27]),
          "+r"(c[28]), "+r"(c[29]), "+r"(c[30]), "+r"(c[31])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
}

union GmmaDescriptor
{
    __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}
    __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept : desc_(t.desc_) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept : desc_(t.desc_) {}

    __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct
    {
        uint16_t start_address_ : 14, : 2;       // 14 bits [0,14), 2 bits unused
        uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
        uint8_t : 1, base_offset_ : 3, : 4;      // 1 bit unused, 3 bits [1,4), 4 bits unused
        uint8_t : 6, layout_type_ : 2;           // 6 bits unused, 2 bits [6,8)
    } bitfield;

    // Decay to a uint64_t
    __device__ constexpr
    operator uint64_t() const noexcept { return desc_; }
};

__device__ __forceinline__ GmmaDescriptor make_wgmma_desc(void *smem_ptr, int siwzzle_type, int sbo, int lbo)
{
    GmmaDescriptor desc;
    desc.bitfield.layout_type_ = siwzzle_type;
    desc.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smem_ptr) >> 4);
    desc.bitfield.base_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = sbo;
    desc.bitfield.leading_byte_offset_ = lbo;
    return desc;
}

__device__ __forceinline__ static GmmaDescriptor gemm_desc_offset(GmmaDescriptor &desc_, int offset)
{
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return ret;
}

__device__ __forceinline__ static void
gemm(int M, int N, int K, GmmaDescriptor &desc_a, GmmaDescriptor &desc_b, uint32_t *reg_c, int stage)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                // int is = (j & 1) ? M - 1 - i : i; // Serpentine coordinate
                int offset_a = i * 512 + k * 2 + stage * 1024;
                int offset_b = j * 512 + k * 128 + stage * 1024; // j is always 0
                auto desc_a_offset = gemm_desc_offset(desc_a, offset_a);
                auto desc_b_offset = gemm_desc_offset(desc_b, offset_b);
                fma(desc_a_offset.desc_, desc_b_offset.desc_, reg_c + i * 32);
            }
        }
    }
}

__device__ void warpgroup_arrive()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait()
{
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// make tma desc
template <class T, uint32_t RANK = 2>
CUtensorMap make_gemm_tma_desc(void *gmem_tensor_ptr, std::vector<int> &gmem_shape, std::vector<int> &smem_shape, CUtensorMapSwizzle swizzle)
{
    CUtensorMap tensor_map{};

    uint64_t gmem_prob_shape[5] = {1, 1, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {0, 0, 0, 0, 0};
    uint32_t smem_box_shape[5] = {1, 1, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    gmem_prob_shape[0] = gmem_shape[0];
    gmem_prob_stride[0] = 1;
    smem_box_shape[0] = smem_shape[0];

    for (int i = 1; i < RANK; ++i)
    {
        gmem_prob_shape[i] = gmem_shape[i];
        gmem_prob_stride[i] = gmem_prob_stride[i - 1] * gmem_shape[i - 1];
        smem_box_shape[i] = smem_shape[i];
    }

    for (int i = 0; i < RANK; ++i)
    {
        gmem_prob_stride[i] *= sizeof(T);
    }

    auto smem_swizzle = swizzle;
    auto tma_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    auto tma_interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    auto tma_l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    auto tma_oobFill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Create the tensor descriptor.
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, // CUtensorMap *tensorMap,
        tma_format,
        RANK,                 // cuuint32_t tensorRank,
        gmem_tensor_ptr,      // void *globalAddress,
        gmem_prob_shape,      // const cuuint64_t *globalDim,
        gmem_prob_stride + 1, // const cuuint64_t *globalStrides,
        smem_box_shape,       // const cuuint32_t *boxDim,
        smem_box_stride,      // const cuuint32_t *elementStrides,
        tma_interleave,       // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
        smem_swizzle,         // Swizzling can be used to avoid shared memory bank conflicts.
        tma_l2Promotion,      // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
        tma_oobFill           // Any element that is outside of bounds will be set to zero by the TMA transfer.
    );

    return tensor_map;
}

enum class CacheHintSm90 : uint64_t
{
    EVICT_NORMAL = 0x1000000000000000,
    EVICT_FIRST = 0x12F0000000000000,
    EVICT_LAST = 0x14F0000000000000,
};

__device__ __forceinline__ static void
tma_load_2d(void const *desc_ptr, uint64_t *mbar_ptr, uint64_t cache_hint, void *smem_ptr,
            int32_t const &crd0, int32_t const &crd1)
{
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ static void
tma_copy_a(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 8; // tma box [8, 64]
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (j * row + i) * 8 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}

__device__ __forceinline__ static void
tma_copy_b(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 8;
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (i + j * row) * 8 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}

// mbarrier
__device__ static void mbarrier_init(uint64_t const *smem_ptr, uint32_t arrive_count)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
}

__device__ static void arrive_and_expect_tx(uint64_t const *smem_ptr, uint32_t transaction_bytes)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
}

__device__ static void mbarrier_wait(uint64_t const *smem_ptr, uint32_t phase)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));
}

__device__ static void mbarrier_arrive(uint64_t const *smem_ptr)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
}

__device__ static void
stmatrix_atom(uint32_t const &src0, uint32_t const &src1, uint32_t const &src2, uint32_t const &src3, half *smem_dst)
{
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_dst);
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_int_ptr),
                 "r"(src0), "r"(src1), "r"(src2), "r"(src3));
}

__device__ __forceinline__ static void
stmatrix_copy(uint32_t *frag, half *smem_dst)
{
    int rep = 8;
    int tid = threadIdx.x;
    int warp_idx = canonical_warp_idx_sync();
    int row = tid % 16;
    int col = (tid % 32) / 16;

#pragma unroll
    for (int i = 0; i < rep; ++i)
    {
        uint32_t a0 = frag[i * 4 + 0];
        uint32_t a1 = frag[i * 4 + 1];
        uint32_t a2 = frag[i * 4 + 2];
        uint32_t a3 = frag[i * 4 + 3];

        uint32_t a4 = frag[i * 4 + 0 + 32];
        uint32_t a5 = frag[i * 4 + 1 + 32];
        uint32_t a6 = frag[i * 4 + 2 + 32];
        uint32_t a7 = frag[i * 4 + 3 + 32];
        int offset = row * 128 + (col + i * 2) * 8 + warp_idx * 16 * 128;
        stmatrix_atom(a0, a1, a2, a3, smem_dst + offset);
        stmatrix_atom(a4, a5, a6, a7, smem_dst + offset + 8192);
    }
}

template <uint32_t Stages_>
struct PipelineState
{
    static constexpr uint32_t Stages = Stages_;
    int index_ = 0;
    uint32_t phase_ = 0;
    __device__ void operator++()
    {
        ++index_;
        if (index_ == Stages)
        {
            index_ = 0;
            phase_ ^= 1;
        }
    }
};

template <int bM, int bN, int bK, int NumThreads, class T, class TC, int NumPipe = 1, int base = 0,
          class tmaA, class tmaB>
__global__ void wgmma_tma_kernel(const T *A, const T *B, TC *C, int M, int N, int K, float alpha, float beta,
                                 const __grid_constant__ tmaA tma_a,
                                 const __grid_constant__ tmaB tma_b)
{
    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = elect_one_sync();

    // thread block swizzle
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int y = (oy << base) + (ox & ((1 << base) - 1));
    int x = (ox >> base);

    alignas(128) extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = shared_memory + bM * bK * NumPipe;

    // init mbarrier
    __shared__ alignas(8) uint64_t producer_mbar[NumPipe];
    __shared__ alignas(8) uint64_t consumer_mbar[NumPipe];

    // auto gA = A + x * bM * K;          // A is K-major
    // auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    constexpr int num_box_row_a = bM / 8;
    constexpr int num_box_col_a = bK / 64;
    constexpr int num_box_row_b = bK / 8;
    constexpr int num_box_col_b = bN / 64;

    constexpr int m_size = bM / 64;
    constexpr int n_size = bN / 128;
    constexpr int k_size = bK / 16;

    uint32_t reg_c[64] = {0};

    auto wgmma_desc_a = make_wgmma_desc(sA, 1 /*swizzle type*/, 64 /*sbo*/, 1 /*lbo*/);   // 128B swizzle
    auto wgmma_desc_b = make_wgmma_desc(sB, 1 /*swizzle type*/, 64 /*sbo*/, 512 /*lbo*/); // 128B swizzle

    // tma expect-tx bytes
    constexpr int tma_transaction_bytes = (bM * bK + bN * bK) * sizeof(T);

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile = 0;

#pragma unroll
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        if (warp_idx == 0 && lane_predicate == 1)
        {
            mbarrier_init(&producer_mbar[pipe], 1);
            mbarrier_init(&consumer_mbar[pipe], 128);
        }
    }
    __syncthreads();

    // prefetch
#pragma unroll
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        auto tile_sA = sA + pipe * bM * bK;
        auto tile_sB = sB + pipe * bN * bK;

        if (warp_idx == 0 && lane_predicate == 1)
        {
            arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            tma_copy_a(&tma_a, &producer_mbar[pipe], tile_sA, num_box_row_a, num_box_col_a, k_tile * bK, x * bM);
            tma_copy_b(&tma_b, &producer_mbar[pipe], tile_sB, num_box_row_b, num_box_col_b, y * bN, k_tile * bK);
        }
        ++k_tile;
    }

    PipelineState<NumPipe> read_state;
    PipelineState<NumPipe> write_state;

#pragma unroll
    for (int pipe = 0; pipe < NumPipe - 1; ++pipe)
    {
        mbarrier_wait(&producer_mbar[pipe], read_state.phase_);
        ++read_state;

        warpgroup_arrive();
        gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, pipe);
        warpgroup_commit_batch();
        mbarrier_arrive(&consumer_mbar[pipe]);
    }

    while (k_tile <= k_tile_count)
    {
        warpgroup_wait<1>();
        if (warp_idx == 0 && lane_predicate == 1 && k_tile < k_tile_count)
        {
            int pipe = write_state.index_;
            auto tile_sA = sA + pipe * bM * bK;
            auto tile_sB = sB + pipe * bN * bK;

            mbarrier_wait(&consumer_mbar[pipe], write_state.phase_);
            arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            tma_copy_a(&tma_a, &producer_mbar[pipe], tile_sA, num_box_row_a, num_box_col_a, k_tile * bK, x * bM);
            tma_copy_b(&tma_b, &producer_mbar[pipe], tile_sB, num_box_row_b, num_box_col_b, y * bN, k_tile * bK);
            ++write_state;
        }
        int read_pipe = read_state.index_;
        mbarrier_wait(&producer_mbar[read_pipe], read_state.phase_);

        warpgroup_arrive();
        gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, read_pipe);
        warpgroup_commit_batch();
        mbarrier_arrive(&consumer_mbar[read_pipe]);
        ++read_state;
        ++k_tile;
    }

    warpgroup_wait<0>();

    stmatrix_copy(reg_c, shared_memory);
    __syncthreads();

    int nbN = bN / 8; // use float4
#pragma unroll
    for (int i = threadIdx.x; i < bM * nbN; i += NumThreads)
    {
        int row = i / nbN;
        int col = i % nbN;
        reinterpret_cast<float4 *>(gC + row * N)[col] = reinterpret_cast<float4 *>(shared_memory + row * bN)[col];
    }
}

template <int bM, int bN, int bK, int NumThreads, class T, int NumPipe = 1>
__global__ void gemm_base(const T *A, const T *B, T *C, int M, int N, int K, float alpha, float beta)
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
            float ori_val = static_cast<float>(tC[0]);
            float val_c = 0.0f;
            auto b_ptr = gB + col_in_block;

            for (int k = 0; k < K; ++k)
            {
                val_c += static_cast<float>(tA[k] * b_ptr[k * N]);
            }
            tC[0] = static_cast<T>(val_c * alpha + beta * ori_val);
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 64;
    constexpr int numPipe = 3;

    using T = half;
    using TC = half;

    constexpr int num_threads = 128; // one warpgroup
    int num_blockM = (M + blockM - 1) / blockM;
    int num_blockN = (N + blockN - 1) / blockN;

    constexpr int base = 0;
    num_blockM = num_blockM * (1 << base);
    num_blockN = (num_blockN + (1 << base) - 1) / (1 << base);

    // create tma desc
    std::vector<int> gA_shape = {K, M}; // stride = {1, K}
    std::vector<int> gB_shape = {N, K}; // stride = {1, N}

    // tma copy box is 64×8 for half
    std::vector<int> sA_shape = {64, 8}; // stride = {1, 64}
    std::vector<int> sB_shape = {64, 8}; // stride = {1, 64}

    auto smem_swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    auto tmaA_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(A), gA_shape, sA_shape, smem_swizzle);
    auto tmaB_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(B), gB_shape, sB_shape, smem_swizzle);

    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    auto kernel_fptr = wgmma_tma_kernel<blockM, blockN, blockK, num_threads, T, TC, numPipe, base,
                                        decltype(tmaA_desc), decltype(tmaB_desc)>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta, tmaA_desc, tmaB_desc);
}

/**
 * nvcc wgmma_tma_128BSW.cu -O3 -arch=sm_90a -lcuda -lcublas -o wgmma_tma_128BSW && ./wgmma_tma_128BSW
 * cublas time = 0.178304 ms, TFLPOS = 770.811070, mfu = 0.779384
 * mma time = 0.192716 ms, TFLPOS = 713.166569, mfu = 0.721099
 */
int main()
{
    srand(1234);

    int num = 4096;

    int M = num;
    int N = num;
    int K = num;

    using T = half;  // A, B dtype
    using TC = half; // C dtype

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);
    thrust::host_vector<TC> mma_res(M * N);
    thrust::host_vector<TC> ref_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }
    for (int i = 0; i < N * K; ++i)
    {
        h_C[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<TC> d_C1 = h_C;
    thrust::device_vector<TC> d_C2 = h_C;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = 1.0f, beta = 0.0f;
    // C is column-major
    // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
    //             reinterpret_cast<const __half *>(&alpha),
    //             reinterpret_cast<__half *>(d_A.data().get()), K,
    //             reinterpret_cast<__half *>(d_B.data().get()), N,
    //             reinterpret_cast<const __half *>(&beta),
    //             reinterpret_cast<__half *>(d_C1.data().get()), M);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                 reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                 &beta,
                 reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    ref_res = d_C1;

    solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, 1.0f, 0.0f);
    mma_res = d_C2;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 989e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                         reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                         &beta,
                         reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                         CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C2.data().get(), M, N, K, 1.0f, 0.0f);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}
// (C) Copyright 2000- ECMWF.
// (C) Copyright 2024- NVIDIA.

#ifdef USE_CUTLASS
//#include "hicblas.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#define CUTLASS_CHECK(e)                                                       \
  {                                                                            \
    cutlass::Status err = (e);                                                 \
    if (err != cutlass::Status::kSuccess) {                                    \
      fprintf(stderr, "CUTLASS error: %s, line %d, %s: %i\n", __FILE__,        \
              __LINE__, #e, (int)err);                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#ifdef USE_CUTLASS_3XTF32
constexpr bool use_3xtf32 = true;
#else
constexpr bool use_3xtf32 = false;
#endif


template <typename CutlassGemm>
CutlassGemm &get_cutlass_handle() {
  static auto handle = std::make_unique<CutlassGemm>();
  return *handle;
}

namespace detail {

enum class CutlassType { cutlass_3xtf32, cutlass_fp32 };

template <CutlassType, cublasOperation_t TransA, cublasOperation_t TransB>
class cutlass_sgemm_grouped;

template <cublasOperation_t TransA, cublasOperation_t TransB>
class cutlass_sgemm_grouped<CutlassType::cutlass_3xtf32, TransA, TransB> {
  // this was verified using Ampere and uses 3XTF32
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using MyOp = cutlass::arch::OpMultiplyAddFastF32;

  using Gemm = cutlass::gemm::device::Gemm<
      float,
      std::conditional_t<TransA == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>, //
      float,
      std::conditional_t<TransB == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>, //
      float, cutlass::layout::ColumnMajor,           //
      float,                                         //
      OperatorClass, cutlass::arch::Sm80,            //
      ThreadblockShape, WarpShape, InstructionShape, //
      cutlass::epilogue::thread::LinearCombination<  //
          float,                                     //
          128 / cutlass::sizeof_bits<float>::value,
          float,                                                    //
          float                                                     //
          >,                                                        //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, //
      3,                                                            //
      AlignmentA,                                                   //
      AlignmentB,                                                   //
      true,                                                         //
      MyOp                                                          //
      >;
  // Note that when setting this alignment > 1 the inputs must be properly
  // zero padded, otherwise NaNs might propagate.
  static constexpr int sz_align = 8;

public:
  using real_type = float;
  void operator()(cudaStream_t stream, int m, int n, int k, float alpha,
                  const float *A, int lda, const float *B, int ldb, float beta,
                  float *C, int ldc) const {
    auto &gemm_op = get_cutlass_handle<Gemm>();
    CUTLASS_CHECK(gemm_op(
        {//
         {(m + sz_align - 1) / sz_align * sz_align,
          (n + sz_align - 1) / sz_align * sz_align,
          (k + sz_align - 1) / sz_align * sz_align},
         {const_cast<float *>(A), lda},
         {const_cast<float *>(B), ldb},
         {C, ldc},
         {C, ldc},
         {alpha, beta}},
        nullptr, stream));
  }
};
template <cublasOperation_t TransA, cublasOperation_t TransB>
class cutlass_sgemm_grouped<CutlassType::cutlass_fp32, TransA, TransB> {
  // this was verified using Volta and uses FP32
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OperatorClass = cutlass::arch::OpClassSimt;
  using MyOp = cutlass::arch::OpMultiplyAdd;

  using Gemm = cutlass::gemm::device::Gemm<
      float, //
      std::conditional_t<TransA == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>, //
      float,                                         //
      std::conditional_t<TransB == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>,                //
      float, cutlass::layout::ColumnMajor,                          //
      float,                                                        //
      OperatorClass, cutlass::arch::Sm70,                           //
      ThreadblockShape, WarpShape, InstructionShape,                //
      cutlass::epilogue::thread::LinearCombination<                 //
          float,                                                    //
          1,                                                        //
          float,                                                    //
          float                                                     //
          >,                                                        //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, //
      2,                                                            //
      AlignmentA,                                                   //
      AlignmentB,                                                   //
      true,                                                         //
      MyOp                                                          //
      >;
  // Note that when setting this alignment > 1 the inputs must be properly
  // zero padded, otherwise NaNs might propagate.
  static constexpr int sz_align = 1;

public:
  using real_type = float;
  void operator()(cudaStream_t stream, int m, int n, int k, float alpha,
                  const float *A, int lda, const float *B, int ldb, float beta,
                  float *C, int ldc) const {
    auto &gemm_op = get_cutlass_handle<Gemm>();
    CUTLASS_CHECK(gemm_op(
        {//
         {(m + sz_align - 1) / sz_align * sz_align,
          (n + sz_align - 1) / sz_align * sz_align,
          (k + sz_align - 1) / sz_align * sz_align},
         {const_cast<float *>(A), lda},
         {const_cast<float *>(B), ldb},
         {C, ldc},
         {C, ldc},
         {alpha, beta}},
        nullptr, stream));
  }
};

} // namespace detail
template <cublasOperation_t TransA, cublasOperation_t TransB>
void cutlass_sgemm_wrapper_grouped_op(int resol_id, int blas_id, int m, const int *n, const int *k,
                                      float alpha, const float *A, int lda,
                                      const int64_t *offsetsA, const float *B, const int *ldb,
                                      const int64_t *offsetsB, float beta, float *C,
                                      int ldc, const int64_t *offsetsC, int batchCount,
                                      cudaStream_t stream,
                                      void *growing_allocator) {
  using namespace detail;
  int device;
  HIC_CHECK(cudaGetDevice(&device));
  int capability_major;
  HIC_CHECK(cudaDeviceGetAttribute(&capability_major,
                                    cudaDevAttrComputeCapabilityMajor, device));
  const auto key = make_cache_key(
      resol_id, blas_id, static_cast<int>(TransA), static_cast<int>(TransB), m,
      n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
      batchCount);
  if (capability_major >= 8 && use_3xtf32)
    run_group_graph(cutlass_sgemm_grouped<detail::CutlassType::cutlass_3xtf32,
                                          TransA, TransB>(),
                    key, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
                    ldc, offsetsC, batchCount, stream,
                    growing_allocator);
  else
    run_group_graph(cutlass_sgemm_grouped<detail::CutlassType::cutlass_fp32,
                                          TransA, TransB>(),
                    key, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
                    ldc, offsetsC, batchCount, stream,
                    growing_allocator);
}

void cutlass_sgemm_wrapper_grouped(int resol_id, int blas_id, char transa, char transb,
                                   int m, const int *n, const int *k, float alpha,
                                   const float *A, int lda, const int64_t *offsetsA,
                                   const float *B, const int *ldb, const int64_t *offsetsB, float beta,
                                   float *C, int ldc, const int64_t *offsetsC,
                                   int batchCount, cudaStream_t stream,
                                   void *growing_allocator) {

  if (transa == 'N' && transb == 'N')
    cutlass_sgemm_wrapper_grouped_op<CUBLAS_OP_N, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
        ldc, offsetsC, batchCount, stream, growing_allocator);
  else if (transa == 'N' && transb == 'T')
    cutlass_sgemm_wrapper_grouped_op<CUBLAS_OP_N, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
        ldc, offsetsC, batchCount, stream, growing_allocator);
  else if (transa == 'T' && transb == 'N')
    cutlass_sgemm_wrapper_grouped_op<CUBLAS_OP_T, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
        ldc, offsetsC, batchCount, stream, growing_allocator);
  else if (transa == 'T' && transb == 'T')
    cutlass_sgemm_wrapper_grouped_op<CUBLAS_OP_T, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C,
        ldc, offsetsC, batchCount, stream, growing_allocator);
  else
    assert(false);
}

namespace detail {

template <typename T> class cutlass_device_array {
public:
  cutlass_device_array() = default;

  ~cutlass_device_array() {
    if (data_ == nullptr)
      return;
    int current_device;
    HIC_CHECK(hipGetDevice(&current_device));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(device_));
    HIC_CHECK(hipFree(data_));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(current_device));
  }

  cutlass_device_array(const cutlass_device_array &) = delete;
  cutlass_device_array &operator=(const cutlass_device_array &) = delete;

  void assign(const std::vector<T> &values) {
    if (values.empty())
      return;
    HIC_CHECK(hipGetDevice(&device_));
    HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data_),
                        values.size() * sizeof(T)));
    HIC_CHECK(hipMemcpy(data_, values.data(), values.size() * sizeof(T),
                        hipMemcpyHostToDevice));
  }

  T *get() const { return data_; }

private:
  T *data_ = nullptr;
  int device_ = 0;
};

template <cublasOperation_t TransA, cublasOperation_t TransB>
class cutlass_dgemm_grouped_entry {
  using LayoutA =
      std::conditional_t<TransA == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>;
  using LayoutB =
      std::conditional_t<TransB == CUBLAS_OP_N, cutlass::layout::ColumnMajor,
                         cutlass::layout::RowMajor>;
  using LayoutC = cutlass::layout::ColumnMajor;
  using GemmKernel =
      typename cutlass::gemm::kernel::DefaultGemmGrouped<
          double, LayoutA, cutlass::ComplexTransform::kNone, 1, double,
          LayoutB, cutlass::ComplexTransform::kNone, 1, double, LayoutC,
          double, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<64, 64, 16>,
          cutlass::gemm::GemmShape<32, 32, 16>,
          cutlass::gemm::GemmShape<8, 8, 4>,
          cutlass::epilogue::thread::LinearCombination<double, 1, double,
                                                        double>,
          cutlass::gemm::threadblock::
              GemmBatchedIdentityThreadblockSwizzle,
          4, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

public:
  using real_type = double;

  cutlass_dgemm_grouped_entry(
      int m, const int *n, const int *k, double alpha, const double *A,
      int lda, const int64_t *offsetsA, const double *B, const int *ldb,
      const int64_t *offsetsB, double beta, double *C, int ldc,
      const int64_t *offsetsC, int batchCount, hipStream_t stream) {
    std::vector<double *> ptr_a;
    std::vector<double *> ptr_b;
    std::vector<double *> ptr_c;
    std::vector<int64_t> lda_values;
    std::vector<int64_t> ldb_values;
    std::vector<int64_t> ldc_values;

    host_problem_sizes_.reserve(batchCount);
    ptr_a.reserve(batchCount);
    ptr_b.reserve(batchCount);
    ptr_c.reserve(batchCount);
    lda_values.reserve(batchCount);
    ldb_values.reserve(batchCount);
    ldc_values.reserve(batchCount);

    for (int i = 0; i < batchCount; ++i) {
      if (m == 0 || n[i] == 0 || k[i] == 0)
        continue;
      host_problem_sizes_.emplace_back(m, n[i], k[i]);
      ptr_a.push_back(const_cast<double *>(A) + offsetsA[i]);
      ptr_b.push_back(const_cast<double *>(B) + offsetsB[i]);
      ptr_c.push_back(C + offsetsC[i]);
      lda_values.push_back(lda);
      ldb_values.push_back(ldb[i]);
      ldc_values.push_back(ldc);
    }

    problem_count_ = static_cast<int>(host_problem_sizes_.size());
    problem_sizes_.assign(host_problem_sizes_);
    ptr_a_.assign(ptr_a);
    ptr_b_.assign(ptr_b);
    ptr_c_.assign(ptr_c);
    lda_.assign(lda_values);
    ldb_.assign(ldb_values);
    ldc_.assign(ldc_values);

    threadblock_count_ =
        Gemm::sufficient(host_problem_sizes_.data(), problem_count_);
    if (threadblock_count_ <= 0) {
      std::fprintf(stderr,
                   "CUTLASS grouped DGEMM has no executable threadblocks\n");
      std::exit(EXIT_FAILURE);
    }

    typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
    typename Gemm::Arguments arguments(
        problem_sizes_.get(), problem_count_, threadblock_count_, epilogue,
        ptr_a_.get(), ptr_b_.get(), ptr_c_.get(), ptr_c_.get(), lda_.get(),
        ldb_.get(), ldc_.get(), ldc_.get(), host_problem_sizes_.data());
    CUTLASS_CHECK(Gemm::can_implement(arguments));
    CUTLASS_CHECK(gemm_.initialize(arguments, nullptr, stream));
  }

  int problem_count() const { return problem_count_; }
  int threadblock_count() const { return threadblock_count_; }

  void run(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(launch_mutex_);
    CUTLASS_CHECK(gemm_.run(stream));
  }

private:
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes_;
  cutlass_device_array<cutlass::gemm::GemmCoord> problem_sizes_;
  cutlass_device_array<double *> ptr_a_;
  cutlass_device_array<double *> ptr_b_;
  cutlass_device_array<double *> ptr_c_;
  cutlass_device_array<int64_t> lda_;
  cutlass_device_array<int64_t> ldb_;
  cutlass_device_array<int64_t> ldc_;
  Gemm gemm_;
  int problem_count_ = 0;
  int threadblock_count_ = 0;
  std::mutex launch_mutex_;
};

} // namespace detail

template <typename Entry> struct cutlass_dgemm_grouped_cache_state {
  std::mutex mutex;
  std::unordered_map<cache_key<double>, std::shared_ptr<Entry>,
                     cache_key_hash<double>>
      entries;
};

template <typename Entry>
cutlass_dgemm_grouped_cache_state<Entry> &
get_cutlass_dgemm_grouped_cache_state() {
  static cutlass_dgemm_grouped_cache_state<Entry> state;
  return state;
}

template <typename Entry>
void free_cutlass_dgemm_grouped_cache(void *, size_t) {
  auto &state = get_cutlass_dgemm_grouped_cache_state<Entry>();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.entries.clear();
}

template <typename Entry>
void erase_cutlass_dgemm_grouped_cache(int resol_id) {
  auto &state = get_cutlass_dgemm_grouped_cache_state<Entry>();
  std::lock_guard<std::mutex> lock(state.mutex);
  erase_resol_from_cache(state.entries, resol_id);
}

template <cublasOperation_t TransA, cublasOperation_t TransB>
void cutlass_dgemm_wrapper_grouped_op(
    int resol_id, int blas_id, int m, const int *n, const int *k, double alpha,
    const double *A, int lda, const int64_t *offsetsA, const double *B,
    const int *ldb, const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, hipStream_t stream,
    void *growing_allocator, bool synchronize) {
  using Entry = detail::cutlass_dgemm_grouped_entry<TransA, TransB>;
  const auto key = make_cache_key(
      resol_id, blas_id, static_cast<int>(TransA), static_cast<int>(TransB), m,
      n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
      batchCount);

  {
    std::lock_guard<std::mutex> registration_lock(
        growing_allocator_registration_mutex);
    growing_allocator_register_free_c(
        growing_allocator, free_cutlass_dgemm_grouped_cache<Entry>);
  }

  auto &state = get_cutlass_dgemm_grouped_cache_state<Entry>();
  std::shared_ptr<Entry> entry;
  bool cache_miss = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto found = state.entries.find(key);
    cache_miss = found == state.entries.end();
    if (cache_miss) {
      entry = std::make_shared<Entry>(m, n, k, alpha, A, lda, offsetsA, B, ldb,
                                      offsetsB, beta, C, ldc, offsetsC,
                                      batchCount, stream);
      state.entries.emplace(key, entry);
    } else {
      entry = found->second;
    }
  }

  if (graph_debug_enabled()) {
    std::fprintf(stderr,
                 "EC_CUTLASS_GROUPED_DP event=run rank=%s resol=%d blas=%d "
                 "groups=%d threadblocks=%d cache=%s\n",
                 graph_debug_rank(), resol_id, blas_id, entry->problem_count(),
                 entry->threadblock_count(), cache_miss ? "miss" : "hit");
    std::fflush(stderr);
  }
  entry->run(stream);
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
}

void cutlass_dgemm_wrapper_grouped_true(
    int resol_id, int blas_id, cublasOperation_t transa,
    cublasOperation_t transb, int m, const int *n, const int *k, double alpha,
    const double *A, int lda, const int64_t *offsetsA, const double *B,
    const int *ldb, const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, hipStream_t stream,
    void *growing_allocator, bool synchronize) {
  if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_N)
    cutlass_dgemm_wrapper_grouped_op<CUBLAS_OP_N, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_T)
    cutlass_dgemm_wrapper_grouped_op<CUBLAS_OP_N, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_N)
    cutlass_dgemm_wrapper_grouped_op<CUBLAS_OP_T, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_T)
    cutlass_dgemm_wrapper_grouped_op<CUBLAS_OP_T, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else
    assert(false);
}
//}

#endif

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

  void allocate(std::size_t size) {
    if (size == 0)
      return;
    HIC_CHECK(hipGetDevice(&device_));
    HIC_CHECK(
        hipMalloc(reinterpret_cast<void **>(&data_), size * sizeof(T)));
    size_ = size;
  }

  void assign(const std::vector<T> &values) {
    if (values.empty())
      return;
    HIC_CHECK(hipGetDevice(&device_));
    HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data_),
                        values.size() * sizeof(T)));
    HIC_CHECK(hipMemcpy(data_, values.data(), values.size() * sizeof(T),
                        hipMemcpyHostToDevice));
    size_ = values.size();
  }

  void update(const std::vector<T> &values, hipStream_t stream) {
    assert(values.size() == size_);
    if (values.empty())
      return;
    HIC_CHECK(cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream));
  }

  T *get() const { return data_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
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
          cutlass::gemm::GemmShape<16, 128, 16>,
          cutlass::gemm::GemmShape<16, 32, 16>,
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
      output_offsets_.push_back(offsetsC[i]);
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

  void update_output_base(double *C, hipStream_t stream) {
    host_ptr_c_.clear();
    host_ptr_c_.reserve(output_offsets_.size());
    for (int64_t offset : output_offsets_)
      host_ptr_c_.push_back(C + offset);
    ptr_c_.update(host_ptr_c_, stream);
  }

  void run(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(launch_mutex_);
    CUTLASS_CHECK(gemm_.run(stream));
  }

private:
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes_;
  std::vector<int64_t> output_offsets_;
  std::vector<double *> host_ptr_c_;
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

class cutlass_dgemm_ordered_scratch_pool {
public:
  cutlass_dgemm_ordered_scratch_pool() {
    HIC_CHECK(hipGetDevice(&device_));
    HIC_CHECK(
        hipEventCreateWithFlags(&completion_event_, hipEventDisableTiming));
  }

  ~cutlass_dgemm_ordered_scratch_pool() {
    int current_device;
    HIC_CHECK(hipGetDevice(&current_device));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(device_));
    if (launched_)
      HIC_CHECK(cudaEventSynchronize(completion_event_));
    HIC_CHECK(hipEventDestroy(completion_event_));
    if (data_ != nullptr)
      HIC_CHECK(hipFree(data_));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(current_device));
  }

  void ensure_locked(int64_t elements) {
    if (elements <= elements_)
      return;
    if (launched_)
      HIC_CHECK(cudaEventSynchronize(completion_event_));
    if (data_ != nullptr)
      HIC_CHECK(hipFree(data_));
    HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data_),
                        static_cast<std::size_t>(elements) * sizeof(double)));
    elements_ = elements;
    launched_ = false;
  }

  void wait_locked(hipStream_t stream) {
    if (launched_)
      HIC_CHECK(hipStreamWaitEvent(stream, completion_event_, 0));
  }

  void record_locked(hipStream_t stream) {
    HIC_CHECK(hipEventRecord(completion_event_, stream));
    launched_ = true;
  }

  double *data() const { return data_; }
  std::mutex &mutex() { return mutex_; }

private:
  double *data_ = nullptr;
  int64_t elements_ = 0;
  int device_ = 0;
  bool launched_ = false;
  hipEvent_t completion_event_{};
  std::mutex mutex_;
};

inline std::shared_ptr<cutlass_dgemm_ordered_scratch_pool>
get_cutlass_dgemm_ordered_scratch_pool() {
  static std::mutex pools_mutex;
  static std::unordered_map<
      int, std::weak_ptr<cutlass_dgemm_ordered_scratch_pool>>
      pools;
  int device;
  HIC_CHECK(hipGetDevice(&device));
  std::lock_guard<std::mutex> lock(pools_mutex);
  auto pool = pools[device].lock();
  if (!pool) {
    pool = std::make_shared<cutlass_dgemm_ordered_scratch_pool>();
    pools[device] = pool;
  }
  return pool;
}

__global__ void cutlass_dgemm_ordered_reduce_kernel(
    const double *partial, double *output, const int64_t *target_offsets,
    const int64_t *contribution_starts, const int *contribution_counts,
    const int64_t *partial_offsets, int m, int output_columns) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t element_count =
      static_cast<int64_t>(m) * output_columns;
  if (index >= element_count)
    return;

  const int row = static_cast<int>(index % m);
  const int column = static_cast<int>(index / m);
  const int64_t target = target_offsets[column] + row;
  const int64_t begin = contribution_starts[column];
  const int count = contribution_counts[column];
  double value = output[target];
  for (int contribution = 0; contribution < count; ++contribution)
    value += partial[partial_offsets[begin + contribution] + row];
  output[target] = value;
}

template <cublasOperation_t TransA, cublasOperation_t TransB>
class cutlass_dgemm_grouped_ordered_entry {
  using grouped_entry = cutlass_dgemm_grouped_entry<TransA, TransB>;

public:
  cutlass_dgemm_grouped_ordered_entry(
      int m, const int *n, const int *k, double alpha, const double *A,
      int lda, const int64_t *offsetsA, const double *B, const int *ldb,
      const int64_t *offsetsB, double beta, double *C, int ldc,
      const int64_t *offsetsC, int batchCount, hipStream_t stream)
      : m_(m), output_(C) {
    const char *layered_value =
        std::getenv("ECTRANS_GPU_CUTLASS_ORDERED_LAYERED");
    layered_ = layered_value != nullptr && layered_value[0] != '\0' &&
               std::strcmp(layered_value, "0") != 0 && m == ldc;
    if (layered_) {
      for (int group = 0; group < batchCount; ++group) {
        if (m == 0 || n[group] == 0 || k[group] == 0)
          continue;
        if (offsetsC[group] % ldc != 0) {
          layered_ = false;
          break;
        }
      }
    }

    if (layered_) {
      construct_layers(m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
                       beta, C, ldc, offsetsC, batchCount, stream);
      return;
    }

    std::vector<int64_t> scratch_offsets(batchCount, 0);
    std::map<int64_t, std::vector<int64_t>> contributions;
    int64_t partial_elements = 0;

    for (int group = 0; group < batchCount; ++group) {
      if (m == 0 || n[group] == 0 || k[group] == 0)
        continue;
      scratch_offsets[group] = partial_elements;
      for (int column = 0; column < n[group]; ++column) {
        contributions[offsetsC[group] + static_cast<int64_t>(column) * ldc]
            .push_back(partial_elements + static_cast<int64_t>(column) * m);
      }
      partial_elements += static_cast<int64_t>(m) * n[group];
    }

    pool_ = get_cutlass_dgemm_ordered_scratch_pool();
    std::vector<int64_t> target_offsets;
    std::vector<int64_t> contribution_starts;
    std::vector<int> contribution_counts;
    std::vector<int64_t> partial_offsets;
    target_offsets.reserve(contributions.size());
    contribution_starts.reserve(contributions.size());
    contribution_counts.reserve(contributions.size());
    for (const auto &item : contributions) {
      target_offsets.push_back(item.first);
      contribution_starts.push_back(partial_offsets.size());
      contribution_counts.push_back(static_cast<int>(item.second.size()));
      partial_offsets.insert(partial_offsets.end(), item.second.begin(),
                             item.second.end());
    }
    output_columns_ = static_cast<int>(target_offsets.size());
    target_offsets_.assign(target_offsets);
    contribution_starts_.assign(contribution_starts);
    contribution_counts_.assign(contribution_counts);
    partial_offsets_.assign(partial_offsets);

    {
      std::lock_guard<std::mutex> lock(pool_->mutex());
      pool_->ensure_locked(partial_elements);
      gemm_ = std::make_unique<grouped_entry>(
          m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, 0.0,
          pool_->data(), m, scratch_offsets.data(), batchCount, stream);
    }
    partial_elements_ = partial_elements;
    problem_count_ = gemm_->problem_count();
    threadblock_count_ = gemm_->threadblock_count();
  }

  int problem_count() const { return problem_count_; }
  int threadblock_count() const { return threadblock_count_; }
  int64_t partial_elements() const { return partial_elements_; }
  int layer_count() const {
    return layered_ ? static_cast<int>(layers_.size()) : 1;
  }
  const char *mode() const { return layered_ ? "layered" : "partial"; }

  void run(hipStream_t stream) {
    if (layered_) {
      for (auto &layer : layers_)
        layer->run(stream);
      return;
    }

    std::lock_guard<std::mutex> lock(pool_->mutex());
    pool_->ensure_locked(partial_elements_);
    pool_->wait_locked(stream);
    gemm_->update_output_base(pool_->data(), stream);
    gemm_->run(stream);
    const int64_t element_count =
        static_cast<int64_t>(m_) * output_columns_;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((element_count + threads - 1) / threads);
    cutlass_dgemm_ordered_reduce_kernel<<<blocks, threads, 0, stream>>>(
        pool_->data(), output_, target_offsets_.get(),
        contribution_starts_.get(), contribution_counts_.get(),
        partial_offsets_.get(), m_, output_columns_);
    HIC_CHECK(cudaGetLastError());
    pool_->record_locked(stream);
  }

private:
  void construct_layers(
      int m, const int *n, const int *k, double alpha, const double *A,
      int lda, const int64_t *offsetsA, const double *B, const int *ldb,
      const int64_t *offsetsB, double beta, double *C, int ldc,
      const int64_t *offsetsC, int batchCount, hipStream_t stream) {
    std::unordered_map<int64_t, int> last_layer;
    std::vector<int> group_layers(batchCount, -1);
    int layer_count = 0;

    for (int group = 0; group < batchCount; ++group) {
      if (m == 0 || n[group] == 0 || k[group] == 0)
        continue;
      const int64_t first_column = offsetsC[group] / ldc;
      int layer = 0;
      for (int column = 0; column < n[group]; ++column) {
        const auto found = last_layer.find(first_column + column);
        if (found != last_layer.end())
          layer = std::max(layer, found->second + 1);
      }
      group_layers[group] = layer;
      layer_count = std::max(layer_count, layer + 1);
      for (int column = 0; column < n[group]; ++column)
        last_layer[first_column + column] = layer;
    }

    std::vector<std::vector<int>> groups_by_layer(layer_count);
    for (int group = 0; group < batchCount; ++group) {
      if (group_layers[group] >= 0)
        groups_by_layer[group_layers[group]].push_back(group);
    }

    layers_.reserve(layer_count);
    for (const auto &groups : groups_by_layer) {
      std::vector<int> layer_n;
      std::vector<int> layer_k;
      std::vector<int> layer_ldb;
      std::vector<int64_t> layer_offsets_a;
      std::vector<int64_t> layer_offsets_b;
      std::vector<int64_t> layer_offsets_c;
      layer_n.reserve(groups.size());
      layer_k.reserve(groups.size());
      layer_ldb.reserve(groups.size());
      layer_offsets_a.reserve(groups.size());
      layer_offsets_b.reserve(groups.size());
      layer_offsets_c.reserve(groups.size());
      for (int group : groups) {
        layer_n.push_back(n[group]);
        layer_k.push_back(k[group]);
        layer_ldb.push_back(ldb[group]);
        layer_offsets_a.push_back(offsetsA[group]);
        layer_offsets_b.push_back(offsetsB[group]);
        layer_offsets_c.push_back(offsetsC[group]);
      }
      auto entry = std::make_unique<grouped_entry>(
          m, layer_n.data(), layer_k.data(), alpha, A, lda,
          layer_offsets_a.data(), B, layer_ldb.data(), layer_offsets_b.data(),
          beta, C, ldc, layer_offsets_c.data(),
          static_cast<int>(groups.size()), stream);
      problem_count_ += entry->problem_count();
      threadblock_count_ += entry->threadblock_count();
      layers_.push_back(std::move(entry));
    }
  }

  int m_ = 0;
  int output_columns_ = 0;
  int64_t partial_elements_ = 0;
  int problem_count_ = 0;
  int threadblock_count_ = 0;
  double *output_ = nullptr;
  bool layered_ = false;
  std::shared_ptr<cutlass_dgemm_ordered_scratch_pool> pool_;
  cutlass_device_array<int64_t> target_offsets_;
  cutlass_device_array<int64_t> contribution_starts_;
  cutlass_device_array<int> contribution_counts_;
  cutlass_device_array<int64_t> partial_offsets_;
  std::unique_ptr<grouped_entry> gemm_;
  std::vector<std::unique_ptr<grouped_entry>> layers_;
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
                 "m=%d groups=%d threadblocks=%d cache=%s\n",
                 graph_debug_rank(), resol_id, blas_id, m,
                 entry->problem_count(), entry->threadblock_count(),
                 cache_miss ? "miss" : "hit");
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

template <cublasOperation_t TransA, cublasOperation_t TransB>
void cutlass_dgemm_wrapper_grouped_ordered_op(
    int resol_id, int blas_id, int m, const int *n, const int *k, double alpha,
    const double *A, int lda, const int64_t *offsetsA, const double *B,
    const int *ldb, const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, hipStream_t stream,
    void *growing_allocator, bool synchronize) {
  using Entry =
      detail::cutlass_dgemm_grouped_ordered_entry<TransA, TransB>;
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
      entry = std::make_shared<Entry>(m, n, k, alpha, A, lda, offsetsA, B,
                                      ldb, offsetsB, beta, C, ldc, offsetsC,
                                      batchCount, stream);
      state.entries.emplace(key, entry);
    } else {
      entry = found->second;
    }
  }

  if (graph_debug_enabled()) {
    std::fprintf(
        stderr,
        "EC_CUTLASS_ORDERED_DP event=run rank=%s resol=%d blas=%d "
        "m=%d groups=%d threadblocks=%d mode=%s layers=%d partial_mib=%.3f "
        "cache=%s\n",
        graph_debug_rank(), resol_id, blas_id, m, entry->problem_count(),
        entry->threadblock_count(), entry->mode(), entry->layer_count(),
        static_cast<double>(entry->partial_elements()) * sizeof(double) /
            (1024.0 * 1024.0),
        cache_miss ? "miss" : "hit");
    std::fflush(stderr);
  }
  entry->run(stream);
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
}

void cutlass_dgemm_wrapper_grouped_ordered_true(
    int resol_id, int blas_id, cublasOperation_t transa,
    cublasOperation_t transb, int m, const int *n, const int *k, double alpha,
    const double *A, int lda, const int64_t *offsetsA, const double *B,
    const int *ldb, const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, hipStream_t stream,
    void *growing_allocator, bool synchronize) {
  if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_N)
    cutlass_dgemm_wrapper_grouped_ordered_op<CUBLAS_OP_N, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_T)
    cutlass_dgemm_wrapper_grouped_ordered_op<CUBLAS_OP_N, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_N)
    cutlass_dgemm_wrapper_grouped_ordered_op<CUBLAS_OP_T, CUBLAS_OP_N>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_T)
    cutlass_dgemm_wrapper_grouped_ordered_op<CUBLAS_OP_T, CUBLAS_OP_T>(
        resol_id, blas_id, m, n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB,
        beta, C, ldc, offsetsC, batchCount, stream, growing_allocator,
        synchronize);
  else
    assert(false);
}
//}

#endif

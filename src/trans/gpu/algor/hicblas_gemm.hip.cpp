// (C) Copyright 2000- ECMWF.
// (C) Copyright 2024- NVIDIA.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#include "hicblas.h"
#include "hicgraph.h"
#ifdef USE_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#endif

#include "growing_allocator.h"

namespace {
using graph_debug_clock = std::chrono::steady_clock;

std::atomic<uint64_t> graph_debug_sequence{0};
std::mutex growing_allocator_registration_mutex;

bool graph_debug_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GEMM_GRAPH_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool graph_debug_force_group() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GEMM_GRAPH_MODE");
    return value != nullptr && std::strcmp(value, "group") == 0;
  }();
  return enabled;
}

bool graph_debug_force_sync_async() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GEMM_DEBUG_SYNC_ASYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool dgemm_bucketed_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_DGEMM_BUCKETED");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool dgemm_bucketed_graph_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_DGEMM_BUCKETED_GRAPH");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool dgemm_bucketed_selected(int blas_id) {
  static const int scope = [] {
    const char *value = std::getenv("ECTRANS_GPU_DGEMM_BUCKETED_SCOPE");
    if (value != nullptr && std::strcmp(value, "inverse") == 0)
      return 1;
    if (value != nullptr && std::strcmp(value, "direct") == 0)
      return 2;
    return 0;
  }();
  if (scope == 1)
    return blas_id == 11 || blas_id == 12;
  if (scope == 2)
    return blas_id == 21 || blas_id == 22;
  return true;
}

bool dgemm_align8_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_DGEMM_ALIGN8");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool dgemm_standard_legendre_id(int blas_id) {
  return blas_id == 11 || blas_id == 12 || blas_id == 21 || blas_id == 22;
}

void align_standard_legendre_shapes(int blas_id, const int *&n, const int *&k,
                                    int batchCount,
                                    std::vector<int> &aligned_n,
                                    std::vector<int> &aligned_k) {
  if (!dgemm_align8_enabled() || !dgemm_standard_legendre_id(blas_id))
    return;
  aligned_n.assign(n, n + batchCount);
  aligned_k.assign(k, k + batchCount);
  for (int i = 0; i < batchCount; ++i) {
    if (aligned_n[i] > 0 && aligned_k[i] > 0) {
      aligned_n[i] = (aligned_n[i] + 7) & ~7;
      aligned_k[i] = (aligned_k[i] + 7) & ~7;
    }
  }
  n = aligned_n.data();
  k = aligned_k.data();
}

bool dgemm_quantization_diagnostics_enabled() {
  static const bool enabled = [] {
    const char *value =
        std::getenv("ECTRANS_GPU_DGEMM_QUANTIZATION_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool cutlass_grouped_dp_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_CUTLASS_GROUPED_DP");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool cutlass_host_schedule_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_CUTLASS_HOST_SCHEDULE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool cutlass_ordered_dp_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GPU_CUTLASS_ORDERED_DP");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool cutlass_grouped_dp_profitable(int m, const int *n, const int *k,
                                   int batchCount) {
  int active = 0;
  int first_n = 0;
  int first_k = 0;
  bool heterogeneous = false;
  for (int i = 0; i < batchCount; ++i) {
    if (m == 0 || n[i] == 0 || k[i] == 0)
      continue;
    if (active == 0) {
      first_n = n[i];
      first_k = k[i];
    } else if (n[i] != first_n || k[i] != first_k) {
      heterogeneous = true;
    }
    ++active;
  }
  return active >= 8 && heterogeneous;
}

const char *graph_debug_rank() {
  const char *rank = std::getenv("OMPI_COMM_WORLD_RANK");
  if (rank == nullptr)
    rank = std::getenv("PMI_RANK");
  if (rank == nullptr)
    rank = std::getenv("SLURM_PROCID");
  return rank == nullptr ? "unknown" : rank;
}

double graph_debug_ms(graph_debug_clock::time_point begin,
                      graph_debug_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename T>
uint64_t graph_debug_hash_value(uint64_t hash, const T &value) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t graph_debug_signature(int m, const int *n, const int *k, int lda,
                               const int64_t *offsetsA, const int *ldb,
                               const int64_t *offsetsB, int ldc,
                               const int64_t *offsetsC, int batchCount) {
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = graph_debug_hash_value(hash, m);
  hash = graph_debug_hash_value(hash, lda);
  hash = graph_debug_hash_value(hash, ldc);
  hash = graph_debug_hash_value(hash, batchCount);
  for (int i = 0; i < batchCount; ++i) {
    hash = graph_debug_hash_value(hash, n[i]);
    hash = graph_debug_hash_value(hash, k[i]);
    hash = graph_debug_hash_value(hash, ldb[i]);
    hash = graph_debug_hash_value(hash, offsetsA[i]);
    hash = graph_debug_hash_value(hash, offsetsB[i]);
    hash = graph_debug_hash_value(hash, offsetsC[i]);
  }
  return hash;
}

int graph_debug_active_groups(int m, const int *n, const int *k,
                              int batchCount) {
  int active = 0;
  for (int i = 0; i < batchCount; ++i)
    if (m != 0 && n[i] != 0 && k[i] != 0)
      ++active;
  return active;
}

template <typename T> class graph_key_array {
public:
  graph_key_array(const T *data, std::size_t size)
      : data_(size == 0 ? nullptr : data), size_(size) {}

  graph_key_array(const graph_key_array &other) { copy_from(other); }

  graph_key_array(graph_key_array &&other) noexcept { move_from(other); }

  graph_key_array &operator=(const graph_key_array &other) {
    if (this != &other)
      copy_from(other);
    return *this;
  }

  graph_key_array &operator=(graph_key_array &&other) noexcept {
    if (this != &other)
      move_from(other);
    return *this;
  }

  bool operator==(const graph_key_array &other) const {
    return size_ == other.size_ &&
           (size_ == 0 || std::equal(data_, data_ + size_, other.data_));
  }

  uint64_t add_to_hash(uint64_t hash) const {
    for (std::size_t i = 0; i < size_; ++i)
      hash = graph_debug_hash_value(hash, data_[i]);
    return hash;
  }

private:
  void copy_from(const graph_key_array &other) {
    size_ = other.size_;
    owned_.clear();
    if (size_ != 0)
      owned_.assign(other.data_, other.data_ + size_);
    data_ = owned_.empty() ? nullptr : owned_.data();
  }

  void move_from(graph_key_array &other) {
    size_ = other.size_;
    if (other.owned_.empty()) {
      owned_.clear();
      if (size_ != 0)
        owned_.assign(other.data_, other.data_ + size_);
    } else {
      owned_ = std::move(other.owned_);
    }
    data_ = owned_.empty() ? nullptr : owned_.data();
  }

  std::vector<T> owned_;
  const T *data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename Real> struct cache_key {
  cache_key(int resol_id_, int blas_id_, int device_id_, int transa_,
            int transb_, int m_, const int *n_, const int *k_, Real alpha_,
            const Real *A_, int lda_, const int64_t *offsetsA_, const Real *B_,
            const int *ldb_, const int64_t *offsetsB_, Real beta_, Real *C_,
            int ldc_, const int64_t *offsetsC_, int batchCount_)
      : resol_id(resol_id_), blas_id(blas_id_), device_id(device_id_),
        transa(transa_), transb(transb_), m(m_), lda(lda_), ldc(ldc_),
        batchCount(batchCount_), alphaBits(real_bits(alpha_)),
        betaBits(real_bits(beta_)), A(A_), B(B_), C(C_),
        n(n_, batchCount_), k(k_, batchCount_), ldb(ldb_, batchCount_),
        offsetsA(offsetsA_, batchCount_), offsetsB(offsetsB_, batchCount_),
        offsetsC(offsetsC_, batchCount_) {
    signature = calculate_signature();
  }

  bool operator==(const cache_key &other) const {
    return resol_id == other.resol_id && blas_id == other.blas_id &&
           device_id == other.device_id && transa == other.transa &&
           transb == other.transb && m == other.m && lda == other.lda &&
           ldc == other.ldc && batchCount == other.batchCount &&
           alphaBits == other.alphaBits && betaBits == other.betaBits &&
           A == other.A && B == other.B && C == other.C && n == other.n &&
           k == other.k && ldb == other.ldb && offsetsA == other.offsetsA &&
           offsetsB == other.offsetsB && offsetsC == other.offsetsC;
  }

  int resol_id;
  int blas_id;
  int device_id;
  int transa;
  int transb;
  int m;
  int lda;
  int ldc;
  int batchCount;
  uint64_t alphaBits;
  uint64_t betaBits;
  const Real *A;
  const Real *B;
  Real *C;
  graph_key_array<int> n;
  graph_key_array<int> k;
  graph_key_array<int> ldb;
  graph_key_array<int64_t> offsetsA;
  graph_key_array<int64_t> offsetsB;
  graph_key_array<int64_t> offsetsC;
  uint64_t signature;

private:
  static uint64_t real_bits(Real value) {
    static_assert(sizeof(Real) <= sizeof(uint64_t));
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
  }

  uint64_t calculate_signature() const {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = graph_debug_hash_value(hash, resol_id);
    hash = graph_debug_hash_value(hash, blas_id);
    hash = graph_debug_hash_value(hash, device_id);
    hash = graph_debug_hash_value(hash, transa);
    hash = graph_debug_hash_value(hash, transb);
    hash = graph_debug_hash_value(hash, m);
    hash = graph_debug_hash_value(hash, lda);
    hash = graph_debug_hash_value(hash, ldc);
    hash = graph_debug_hash_value(hash, batchCount);
    hash = graph_debug_hash_value(hash, alphaBits);
    hash = graph_debug_hash_value(hash, betaBits);
    hash = graph_debug_hash_value(hash, A);
    hash = graph_debug_hash_value(hash, B);
    hash = graph_debug_hash_value(hash, C);
    hash = n.add_to_hash(hash);
    hash = k.add_to_hash(hash);
    hash = ldb.add_to_hash(hash);
    hash = offsetsA.add_to_hash(hash);
    hash = offsetsB.add_to_hash(hash);
    return offsetsC.add_to_hash(hash);
  }
};

template <typename Real> struct cache_key_hash {
  std::size_t operator()(const cache_key<Real> &key) const {
    return static_cast<std::size_t>(key.signature);
  }
};

template <typename Real>
cache_key<Real> make_cache_key(
    int resol_id, int blas_id, int transa, int transb, int m, const int *n,
    const int *k, Real alpha, const Real *A, int lda, const int64_t *offsetsA,
    const Real *B, const int *ldb, const int64_t *offsetsB, Real beta, Real *C,
    int ldc, const int64_t *offsetsC, int batchCount) {
  int device_id;
  HIC_CHECK(hipGetDevice(&device_id));
  return cache_key<Real>{resol_id, blas_id, device_id, transa, transb,
                         m,        n,       k,         alpha,  A,
                         lda,      offsetsA, B,        ldb,    offsetsB,
                         beta,     C,       ldc,       offsetsC,
                         batchCount};
}

class graph_exec_entry {
public:
  explicit graph_exec_entry(hipGraphExec_t graph) : graph_(graph) {}

  ~graph_exec_entry() { HIC_CHECK(hipGraphExecDestroy(graph_)); }

  graph_exec_entry(const graph_exec_entry &) = delete;
  graph_exec_entry &operator=(const graph_exec_entry &) = delete;

  void launch(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(launch_mutex_);
    HIC_CHECK(hipGraphLaunch(graph_, stream));
  }

private:
  hipGraphExec_t graph_;
  std::mutex launch_mutex_;
};

template <typename Gemm> struct graph_cache_state {
  using real_t = typename Gemm::real_type;

  std::mutex mutex;
  std::unordered_map<cache_key<real_t>, std::shared_ptr<graph_exec_entry>,
                     cache_key_hash<real_t>>
      graph_cache;
  std::unordered_map<cache_key<real_t>, bool, cache_key_hash<real_t>>
      warmup_cache;
};

template <typename Gemm> graph_cache_state<Gemm> &get_graph_cache_state() {
  static graph_cache_state<Gemm> state;
  return state;
}

template <typename Gemm> void free_gemm_graph_cache(void *, size_t) {
  auto &state = get_graph_cache_state<Gemm>();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (graph_debug_enabled()) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=cache_clear reason=allocator rank=%s "
                 "pid=%ld precision=%s graph_entries=%zu warmup_entries=%zu\n",
                 graph_debug_rank(), static_cast<long>(getpid()),
                 std::is_same<typename Gemm::real_type, double>::value ? "f64"
                                                                       : "f32",
                 state.graph_cache.size(), state.warmup_cache.size());
    std::fflush(stderr);
  }
  state.graph_cache.clear();
  state.warmup_cache.clear();
}
template <typename Cache>
void erase_resol_from_cache(Cache &cache, int resol_id) {
  // Note that in C++20 this could also be std::erase_if
  int erased = 0;
  for (auto it = cache.begin(); it != cache.end();) {
    if (it->first.resol_id == resol_id) {
      it = cache.erase(it);
      ++erased;
    } else
      ++it;
  }
}
template <typename Gemm> void erase_from_caches(int resol_id) {
  auto &state = get_graph_cache_state<Gemm>();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (graph_debug_enabled()) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=cache_clear reason=resolution rank=%s "
                 "pid=%ld precision=%s resol=%d graph_entries=%zu "
                 "warmup_entries=%zu\n",
                 graph_debug_rank(), static_cast<long>(getpid()),
                 std::is_same<typename Gemm::real_type, double>::value ? "f64"
                                                                       : "f32",
                 resol_id, state.graph_cache.size(),
                 state.warmup_cache.size());
    std::fflush(stderr);
  }
  erase_resol_from_cache(state.graph_cache, resol_id);
  erase_resol_from_cache(state.warmup_cache, resol_id);
}

// this version is using graphs and caches the graphs
template <typename Gemm, typename Real>
void run_group_graph(Gemm &&gemm, const cache_key<Real> &key, int m,
                     const int *n,
                     const int *k, Real alpha, const Real *A, int lda,
                     const int64_t *offsetsA, const Real *B, const int *ldb,
                     const int64_t *offsetsB, Real beta, Real *C, int ldc,
                     const int64_t *offsetsC, int batchCount,
                     hipStream_t stream, void *growing_allocator,
                     bool synchronize = true) {
  const bool debug = graph_debug_enabled();
  const auto callStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  const uint64_t sequence = debug ? ++graph_debug_sequence : 0;
  const uint64_t signature = key.signature;
  const int activeGroups =
      debug ? graph_debug_active_groups(m, n, k, batchCount) : 0;

  {
    std::lock_guard<std::mutex> registration_lock(
        growing_allocator_registration_mutex);
    growing_allocator_register_free_c(growing_allocator,
                                      free_gemm_graph_cache<Gemm>);
  }

  auto &cacheState = get_graph_cache_state<Gemm>();
  std::unique_lock<std::mutex> cacheLock(cacheState.mutex);
  auto &graphCache = cacheState.graph_cache;

  auto graph = graphCache.find(key);
  const bool cacheMiss = graph == graphCache.end();
  std::shared_ptr<graph_exec_entry> graphExec;
  if (!cacheMiss)
    graphExec = graph->second;
  if (debug) {
    std::fprintf(
        stderr,
        "EC_GRAPH_DEBUG event=call mode=graph rank=%s pid=%ld seq=%" PRIu64
        " precision=%s resol=%d m=%d blas=%d device=%d transa=%d transb=%d "
        "batch=%d active=%d signature=%016" PRIx64
        " cache=%s graph_entries=%zu stream=%p A=%p B=%p C=%p\n",
        graph_debug_rank(), static_cast<long>(getpid()), sequence,
        std::is_same<Real, double>::value ? "f64" : "f32", key.resol_id, m,
        key.blas_id, key.device_id, key.transa, key.transb, batchCount,
        activeGroups, signature, cacheMiss ? "miss" : "hit", graphCache.size(),
        reinterpret_cast<void *>(stream), static_cast<const void *>(A),
        static_cast<const void *>(B), static_cast<void *>(C));
    std::fflush(stderr);
  }

  if (graph == graphCache.end()) {
    // this graph does not exist yet
    const auto createStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    hipStream_t captureStream;
    HIC_CHECK(hipStreamCreate(&captureStream));
    const auto createEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    std::size_t graphNodes = 0;
    auto beginStart = createEnd;
    auto beginEnd = createEnd;
    auto enqueueEnd = createEnd;
    auto captureEnd = createEnd;
    auto instantiateEnd = createEnd;

#ifdef USE_CUTLASS
    hipGraph_t new_graph_raw;
    HIC_CHECK(hipGraphCreate(&new_graph_raw, 0));
    hic_graph_owner new_graph{new_graph_raw};
    std::vector<hic_graph_owner> child_graphs;
    child_graphs.reserve(batchCount);
    for (int i = 0; i < batchCount; ++i) {
      if (m == 0 || n[i] == 0 || k[i] == 0)
        continue;

      HIC_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
      gemm(captureStream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
           ldb[i], beta, C + offsetsC[i], ldc);
      hipGraph_t my_graph_raw;
      HIC_CHECK(hipStreamEndCapture(captureStream, &my_graph_raw));
      hic_graph_owner my_graph{my_graph_raw};
      hipGraphNode_t my_node;
      HIC_CHECK(
          hipGraphAddChildGraphNode(&my_node, new_graph.get(), nullptr, 0,
                                    my_graph.get()));
      child_graphs.push_back(std::move(my_graph));
    }
    if (debug)
      HIC_CHECK(hipGraphGetNodes(new_graph.get(), nullptr, &graphNodes));
    const auto instantiateStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, new_graph.get(), NULL, NULL, 0));
    if (debug)
      instantiateEnd = graph_debug_clock::now();
    child_graphs.clear();
    new_graph.reset();
#else
    if (debug)
      beginStart = graph_debug_clock::now();
    HIC_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
    if (debug)
      beginEnd = graph_debug_clock::now();
    for (int i = 0; i < batchCount; ++i) {
      if (m == 0 || n[i] == 0 || k[i] == 0)
        continue;

      gemm(captureStream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
           ldb[i], beta, C + offsetsC[i], ldc);
    }
    if (debug)
      enqueueEnd = graph_debug_clock::now();
    hipGraph_t my_graph_raw;
    HIC_CHECK(hipStreamEndCapture(captureStream, &my_graph_raw));
    hic_graph_owner my_graph{my_graph_raw};
    if (debug) {
      captureEnd = graph_debug_clock::now();
      HIC_CHECK(hipGraphGetNodes(my_graph.get(), nullptr, &graphNodes));
    }
    const auto instantiateStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, my_graph.get(), NULL, NULL, 0));
    if (debug)
      instantiateEnd = graph_debug_clock::now();
    my_graph.reset();
#endif
    const auto destroyStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    HIC_CHECK(hipStreamDestroy(captureStream));
    const auto destroyEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};

    graphExec = std::make_shared<graph_exec_entry>(instance);
    graphCache.emplace(key, graphExec);
    if (debug) {
      std::fprintf(
          stderr,
          "EC_GRAPH_DEBUG event=build rank=%s pid=%ld seq=%" PRIu64
          " precision=%s resol=%d m=%d blas=%d batch=%d active=%d nodes=%zu "
          "stream_create_ms=%.3f begin_capture_ms=%.3f enqueue_ms=%.3f "
          "end_capture_ms=%.3f instantiate_ms=%.3f stream_destroy_ms=%.3f "
          "build_total_ms=%.3f\n",
          graph_debug_rank(), static_cast<long>(getpid()), sequence,
          std::is_same<Real, double>::value ? "f64" : "f32", key.resol_id, m,
          key.blas_id, batchCount, activeGroups, graphNodes,
          graph_debug_ms(createStart, createEnd),
          graph_debug_ms(beginStart, beginEnd),
          graph_debug_ms(beginEnd, enqueueEnd),
          graph_debug_ms(enqueueEnd, captureEnd),
          graph_debug_ms(captureEnd, instantiateEnd),
          graph_debug_ms(destroyStart, destroyEnd),
          graph_debug_ms(createStart, destroyEnd));
      std::fflush(stderr);
    }
  }
  cacheLock.unlock();

  const auto launchStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  graphExec->launch(stream);
  const auto launchEnd =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  auto syncEnd = launchEnd;
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
  if (debug)
    syncEnd = graph_debug_clock::now();
  if (debug) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=return mode=graph rank=%s pid=%ld "
                 "seq=%" PRIu64 " precision=%s resol=%d m=%d blas=%d "
                 "cache=%s launch_ms=%.3f sync_ms=%.3f host_total_ms=%.3f\n",
                 graph_debug_rank(), static_cast<long>(getpid()), sequence,
                 std::is_same<Real, double>::value ? "f64" : "f32",
                 key.resol_id, m, key.blas_id, cacheMiss ? "miss" : "hit",
                 graph_debug_ms(launchStart, launchEnd),
                 graph_debug_ms(launchEnd, syncEnd),
                 graph_debug_ms(callStart, syncEnd));
    std::fflush(stderr);
  }
}

// stupid simple gemm calls
template <typename Gemm, typename Real>
void run_group(Gemm &&gemm, int resol_id, int m, const int *n, const int *k,
               Real alpha, const Real *A, int lda, const int64_t *offsetsA,
               const Real *B, const int *ldb, const int64_t *offsetsB,
               Real beta, Real *C, int ldc, const int64_t *offsetsC,
               int batchCount, hipStream_t stream, int blas_id = -1,
               bool synchronize = true) {
  const bool debug = graph_debug_enabled();
  const auto callStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  const uint64_t sequence = debug ? ++graph_debug_sequence : 0;
  const uint64_t signature =
      debug ? graph_debug_signature(m, n, k, lda, offsetsA, ldb, offsetsB,
                                    ldc, offsetsC, batchCount)
            : 0;
  const int activeGroups =
      debug ? graph_debug_active_groups(m, n, k, batchCount) : 0;
  const auto enqueueStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  for (int i = 0; i < batchCount; ++i) {
    if (m == 0 || n[i] == 0 || k[i] == 0)
      continue;
    gemm(stream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
         ldb[i], beta, C + offsetsC[i], ldc);
  }
  const auto enqueueEnd =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
  const auto syncEnd =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  if (debug) {
    std::fprintf(
        stderr,
        "EC_GRAPH_DEBUG event=return mode=group rank=%s pid=%ld seq=%" PRIu64
        " precision=%s resol=%d m=%d blas=%d batch=%d active=%d "
        "signature=%016" PRIx64 " stream=%p A=%p B=%p C=%p "
        "enqueue_ms=%.3f sync_ms=%.3f host_total_ms=%.3f\n",
        graph_debug_rank(), static_cast<long>(getpid()), sequence,
        std::is_same<Real, double>::value ? "f64" : "f32", resol_id, m,
        blas_id, batchCount, activeGroups, signature,
        reinterpret_cast<void *>(stream), static_cast<const void *>(A),
        static_cast<const void *>(B), static_cast<void *>(C),
        graph_debug_ms(enqueueStart, enqueueEnd),
        graph_debug_ms(enqueueEnd, syncEnd),
        graph_debug_ms(callStart, syncEnd));
    std::fflush(stderr);
  }
}

#ifdef USE_CUTLASS
#include "hicblas_cutlass.cuda.h"
#endif

class hipblas_handle_owner {
public:
  explicit hipblas_handle_owner(int device) : device_(device) {
    HICBLAS_CHECK(hipblasCreate(&handle_));
  }

  ~hipblas_handle_owner() {
    int current_device;
    HIC_CHECK(hipGetDevice(&current_device));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(device_));
    HICBLAS_CHECK(hipblasDestroy(handle_));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(current_device));
  }

  hipblas_handle_owner(const hipblas_handle_owner &) = delete;
  hipblas_handle_owner &operator=(const hipblas_handle_owner &) = delete;

  hipblasHandle_t get() const { return handle_; }

private:
  int device_;
  hipblasHandle_t handle_;
};

hipblasHandle_t get_hipblas_handle() {
  int device;
  HIC_CHECK(hipGetDevice(&device));
  thread_local std::unordered_map<int, std::unique_ptr<hipblas_handle_owner>>
      handles;
  auto handle = handles.find(device);
  if (handle == handles.end()) {
    handle = handles
                 .emplace(device,
                          std::make_unique<hipblas_handle_owner>(device))
                 .first;
  }
  return handle->second->get();
}
template <typename Real> struct hipblas_gemm_grouped {
public:
  using real_type = Real;
  hipblas_gemm_grouped(hipblasOperation_t transa, hipblasOperation_t transb)
      : transa_(transa), transb_(transb) {
    // we need to get the hipblas handle here, otherwise this could be created
    // during graph capturing
    get_hipblas_handle();
  };
  void operator()(hipStream_t stream, int m, int n, int k, Real alpha,
                  const Real *A, int lda, const Real *B, int ldb, Real beta,
                  Real *C, int ldc) const {
    hipblasHandle_t handle = get_hipblas_handle();
    HICBLAS_CHECK(hipblasSetStream(handle, stream));

    if constexpr (std::is_same<Real, float>::value)
      HICBLAS_CHECK(hipblasSgemm(handle, transa_, transb_, m, n, k, &alpha, A,
                                 lda, B, ldb, &beta, C, ldc));
    if constexpr (std::is_same<Real, double>::value)
      HICBLAS_CHECK(hipblasDgemm(handle, transa_, transb_, m, n, k, &alpha, A,
                                 lda, B, ldb, &beta, C, ldc));
  }

private:
  hipblasOperation_t transa_, transb_;
};

struct dgemm_bucket_shape {
  int n;
  int k;
  int ldb;

  bool operator<(const dgemm_bucket_shape &other) const {
    return std::tie(n, k, ldb) < std::tie(other.n, other.k, other.ldb);
  }
};

int dgemm_graph_lane_limit();

class dgemm_bucketed_entry {
public:
  dgemm_bucketed_entry(int device, const double *A,
                       const int64_t *offsetsA, const double *B,
                       const int *n, const int *k, const int *ldb,
                       const int64_t *offsetsB, double *C,
                       const int64_t *offsetsC, int batchCount)
      : device_(device) {
    std::map<dgemm_bucket_shape, std::vector<int>> groups;
    for (int i = 0; i < batchCount; ++i) {
      if (n[i] > 0 && k[i] > 0)
        groups[{n[i], k[i], ldb[i]}].push_back(i);
    }

    buckets_.reserve(groups.size());
    for (const auto &item : groups) {
      bucket data;
      data.shape = item.first;
      data.count = static_cast<int>(item.second.size());
      std::vector<const double *> hostA(data.count);
      std::vector<const double *> hostB(data.count);
      std::vector<double *> hostC(data.count);
      for (int j = 0; j < data.count; ++j) {
        const int index = item.second[j];
        hostA[j] = A + offsetsA[index];
        hostB[j] = B + offsetsB[index];
        hostC[j] = C + offsetsC[index];
      }
      HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data.A),
                          data.count * sizeof(*data.A)));
      HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data.B),
                          data.count * sizeof(*data.B)));
      HIC_CHECK(hipMalloc(reinterpret_cast<void **>(&data.C),
                          data.count * sizeof(*data.C)));
      HIC_CHECK(hipMemcpy(data.A, hostA.data(), data.count * sizeof(*data.A),
                          hipMemcpyHostToDevice));
      HIC_CHECK(hipMemcpy(data.B, hostB.data(), data.count * sizeof(*data.B),
                          hipMemcpyHostToDevice));
      HIC_CHECK(hipMemcpy(data.C, hostC.data(), data.count * sizeof(*data.C),
                          hipMemcpyHostToDevice));
      active_groups_ += data.count;
      largest_bucket_ = std::max(largest_bucket_, data.count);
      buckets_.push_back(data);
    }
  }

  ~dgemm_bucketed_entry() {
    int current_device;
    HIC_CHECK(hipGetDevice(&current_device));
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(device_));
    for (auto &data : buckets_) {
      HIC_CHECK(hipFree(data.A));
      HIC_CHECK(hipFree(data.B));
      HIC_CHECK(hipFree(data.C));
    }
    if (current_device != device_)
      HIC_CHECK(hipSetDevice(current_device));
  }

  dgemm_bucketed_entry(const dgemm_bucketed_entry &) = delete;
  dgemm_bucketed_entry &operator=(const dgemm_bucketed_entry &) = delete;

  void run(hipblasOperation_t transa, hipblasOperation_t transb, int m,
           double alpha, int lda, double beta, int ldc, hipStream_t stream) {
    if (!dgemm_bucketed_graph_enabled()) {
      run_direct(transa, transb, m, alpha, lda, beta, ldc, stream);
      return;
    }

    std::shared_ptr<graph_exec_entry> graph;
    {
      std::lock_guard<std::mutex> lock(graph_mutex_);
      if (!graph_)
        graph_ = build_graph(transa, transb, m, alpha, lda, beta, ldc, stream);
      graph = graph_;
    }
    graph->launch(stream);
  }

  int bucket_count() const { return static_cast<int>(buckets_.size()); }
  int active_groups() const { return active_groups_; }
  int largest_bucket() const { return largest_bucket_; }

private:
  struct bucket {
    dgemm_bucket_shape shape{};
    int count = 0;
    const double **A = nullptr;
    const double **B = nullptr;
    double **C = nullptr;
  };

  static void submit_bucket(hipblasHandle_t handle,
                            hipblasOperation_t transa,
                            hipblasOperation_t transb, int m, double alpha,
                            int lda, double beta, int ldc, hipStream_t stream,
                            const bucket &data) {
    HICBLAS_CHECK(hipblasSetStream(handle, stream));
    HICBLAS_CHECK(hipblasDgemmBatched(
        handle, transa, transb, m, data.shape.n, data.shape.k, &alpha, data.A,
        lda, data.B, data.shape.ldb, &beta, data.C, ldc, data.count));
  }

  void run_direct(hipblasOperation_t transa, hipblasOperation_t transb, int m,
                  double alpha, int lda, double beta, int ldc,
                  hipStream_t stream) {
    hipblasHandle_t handle = get_hipblas_handle();
    for (const auto &data : buckets_)
      submit_bucket(handle, transa, transb, m, alpha, lda, beta, ldc, stream,
                    data);
  }

  std::shared_ptr<graph_exec_entry>
  build_graph(hipblasOperation_t transa, hipblasOperation_t transb, int m,
              double alpha, int lda, double beta, int ldc,
              hipStream_t caller_stream) {
    const int lanes = std::min(dgemm_graph_lane_limit(), bucket_count());
    std::vector<int> order(bucket_count());
    for (int i = 0; i < bucket_count(); ++i)
      order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
      const auto &lhs = buckets_[left];
      const auto &rhs = buckets_[right];
      return static_cast<uint64_t>(lhs.count) * lhs.shape.n * lhs.shape.k >
             static_cast<uint64_t>(rhs.count) * rhs.shape.n * rhs.shape.k;
    });

    std::vector<std::vector<int>> lane_buckets(lanes);
    std::vector<uint64_t> lane_work(lanes, 0);
    for (int index : order) {
      const int lane = static_cast<int>(
          std::min_element(lane_work.begin(), lane_work.end()) -
          lane_work.begin());
      lane_buckets[lane].push_back(index);
      const auto &data = buckets_[index];
      lane_work[lane] +=
          static_cast<uint64_t>(data.count) * data.shape.n * data.shape.k;
    }

    HIC_CHECK(hipStreamSynchronize(caller_stream));
    int device;
    HIC_CHECK(hipGetDevice(&device));
    std::vector<hipStream_t> lane_streams(lanes);
    std::vector<std::unique_ptr<hipblas_handle_owner>> lane_handles;
    lane_handles.reserve(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
      HIC_CHECK(hipStreamCreate(&lane_streams[lane]));
      lane_handles.emplace_back(std::make_unique<hipblas_handle_owner>(device));
      submit_bucket(lane_handles[lane]->get(), transa, transb, m, alpha, lda,
                    beta, ldc, lane_streams[lane],
                    buckets_[lane_buckets[lane].front()]);
      HIC_CHECK(hipStreamSynchronize(lane_streams[lane]));
    }

    hipEvent_t fork_event;
    HIC_CHECK(hipEventCreateWithFlags(&fork_event, hipEventDisableTiming));
    std::vector<hipEvent_t> completion_events(lanes - 1);
    for (hipEvent_t &event : completion_events)
      HIC_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

    HIC_CHECK(hipStreamBeginCapture(lane_streams[0],
                                    hipStreamCaptureModeGlobal));
    HIC_CHECK(hipEventRecord(fork_event, lane_streams[0]));
    for (int lane = 1; lane < lanes; ++lane)
      HIC_CHECK(hipStreamWaitEvent(lane_streams[lane], fork_event, 0));
    for (int lane = 0; lane < lanes; ++lane)
      for (int index : lane_buckets[lane])
        submit_bucket(lane_handles[lane]->get(), transa, transb, m, alpha,
                      lda, beta, ldc, lane_streams[lane], buckets_[index]);
    for (int lane = 1; lane < lanes; ++lane) {
      HIC_CHECK(hipEventRecord(completion_events[lane - 1],
                               lane_streams[lane]));
      HIC_CHECK(hipStreamWaitEvent(lane_streams[0],
                                   completion_events[lane - 1], 0));
    }

    hipGraph_t graph_raw;
    HIC_CHECK(hipStreamEndCapture(lane_streams[0], &graph_raw));
    hic_graph_owner captured_graph{graph_raw};
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, captured_graph.get(), nullptr,
                                  nullptr, 0));

    lane_handles.clear();
    HIC_CHECK(hipEventDestroy(fork_event));
    for (hipEvent_t event : completion_events)
      HIC_CHECK(hipEventDestroy(event));
    for (hipStream_t lane_stream : lane_streams)
      HIC_CHECK(hipStreamDestroy(lane_stream));
    captured_graph.reset();

    if (graph_debug_enabled()) {
      const auto limits =
          std::minmax_element(lane_work.begin(), lane_work.end());
      std::fprintf(stderr,
                   "EC_DGEMM_BUCKETED event=graph_build rank=%s buckets=%d "
                   "lanes=%d lane_work_min=%" PRIu64
                   " lane_work_max=%" PRIu64 "\n",
                   graph_debug_rank(), bucket_count(), lanes, *limits.first,
                   *limits.second);
      std::fflush(stderr);
    }
    return std::make_shared<graph_exec_entry>(instance);
  }

  int device_ = 0;
  int active_groups_ = 0;
  int largest_bucket_ = 0;
  std::vector<bucket> buckets_;
  std::mutex graph_mutex_;
  std::shared_ptr<graph_exec_entry> graph_;
};

struct dgemm_bucketed_cache_state {
  std::mutex mutex;
  std::unordered_map<cache_key<double>, std::shared_ptr<dgemm_bucketed_entry>,
                     cache_key_hash<double>>
      entries;
};

dgemm_bucketed_cache_state &get_dgemm_bucketed_cache_state() {
  static dgemm_bucketed_cache_state state;
  return state;
}

void free_dgemm_bucketed_cache(void *, size_t) {
  auto &state = get_dgemm_bucketed_cache_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.entries.clear();
}

void erase_dgemm_bucketed_cache(int resol_id) {
  auto &state = get_dgemm_bucketed_cache_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  erase_resol_from_cache(state.entries, resol_id);
}

bool dgemm_bucketed_profitable(int m, const int *n, const int *k, int ldc,
                               const int64_t *offsetsC, int batchCount) {
  if (m <= 0 || batchCount < 8)
    return false;
  std::vector<std::pair<int64_t, int64_t>> ranges;
  ranges.reserve(batchCount);
  for (int i = 0; i < batchCount; ++i) {
    if (n[i] <= 0 || k[i] <= 0)
      continue;
    ranges.emplace_back(offsetsC[i], offsetsC[i] +
                                         static_cast<int64_t>(n[i] - 1) * ldc +
                                         m);
  }
  if (ranges.size() < 8)
    return false;
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      return false;
  return true;
}

void diagnose_dgemm_quantization(
    int resol_id, int blas_id, int m, const int *n, const int *k, int lda,
    const int64_t *offsetsA, const int *ldb, const int64_t *offsetsB, int ldc,
    const int64_t *offsetsC, int batchCount) {
  if (!dgemm_quantization_diagnostics_enabled() || m <= 0 || batchCount < 8)
    return;

  const uint64_t signature = graph_debug_signature(
      m, n, k, lda, offsetsA, ldb, offsetsB, ldc, offsetsC, batchCount);
  static std::mutex reported_mutex;
  static std::map<std::tuple<int, int, uint64_t>, bool> reported;
  {
    std::lock_guard<std::mutex> lock(reported_mutex);
    if (!reported.emplace(std::make_tuple(resol_id, blas_id, signature), true)
             .second)
      return;
  }

  std::map<std::pair<int, int>, int> exact;
  long double original_work = 0.0L;
  int active = 0;
  for (int i = 0; i < batchCount; ++i) {
    if (n[i] <= 0 || k[i] <= 0)
      continue;
    ++exact[{n[i], k[i]}];
    original_work += static_cast<long double>(n[i]) * k[i];
    ++active;
  }
  if (active < 8)
    return;

  for (const int quantum : {4, 8, 16, 32}) {
    std::map<std::pair<int, int>, int> k_only;
    std::map<std::pair<int, int>, int> both;
    long double quantized_work = 0.0L;
    int largest = 0;
    for (int i = 0; i < batchCount; ++i) {
      if (n[i] <= 0 || k[i] <= 0)
        continue;
      const int qn = ((n[i] + quantum - 1) / quantum) * quantum;
      const int qk = ((k[i] + quantum - 1) / quantum) * quantum;
      ++k_only[{n[i], qk}];
      largest = std::max(largest, ++both[{qn, qk}]);
      quantized_work += static_cast<long double>(qn) * qk;
    }
    std::fprintf(
        stderr,
        "EC_DGEMM_QUANT rank=%s resol=%d blas=%d groups=%d quantum=%d "
        "exact_buckets=%zu k_buckets=%zu nk_buckets=%zu largest=%d "
        "work_ratio=%.6Lf\n",
        graph_debug_rank(), resol_id, blas_id, active, quantum, exact.size(),
        k_only.size(), both.size(), largest,
        original_work == 0.0L ? 0.0L : quantized_work / original_work);
  }
  std::fflush(stderr);
}

void run_dgemm_bucketed(
    hipblasOperation_t transa, hipblasOperation_t transb,
    const cache_key<double> &key, int m, const int *n, const int *k,
    double alpha, const double *A, int lda, const int64_t *offsetsA,
    const double *B, const int *ldb, const int64_t *offsetsB, double beta,
    double *C, int ldc, const int64_t *offsetsC, int batchCount,
    hipStream_t stream, void *growing_allocator, bool synchronize) {
  {
    std::lock_guard<std::mutex> registration_lock(
        growing_allocator_registration_mutex);
    growing_allocator_register_free_c(growing_allocator,
                                      free_dgemm_bucketed_cache);
  }

  auto &state = get_dgemm_bucketed_cache_state();
  std::shared_ptr<dgemm_bucketed_entry> entry;
  bool cache_miss = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto found = state.entries.find(key);
    cache_miss = found == state.entries.end();
    if (cache_miss) {
      entry = std::make_shared<dgemm_bucketed_entry>(
          key.device_id, A, offsetsA, B, n, k, ldb, offsetsB, C, offsetsC,
          batchCount);
      state.entries.emplace(key, entry);
    } else {
      entry = found->second;
    }
  }

  if (graph_debug_enabled() && cache_miss) {
    std::fprintf(stderr,
                 "EC_DGEMM_BUCKETED event=build rank=%s resol=%d blas=%d "
                 "groups=%d buckets=%d largest=%d\n",
                 graph_debug_rank(), key.resol_id, key.blas_id,
                 entry->active_groups(), entry->bucket_count(),
                 entry->largest_bucket());
    std::fflush(stderr);
  }
  entry->run(transa, transb, m, alpha, lda, beta, ldc, stream);
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
}

int dgemm_graph_lane_limit() {
  static const int lanes = [] {
    const char *value = std::getenv("ECTRANS_GEMM_GRAPH_LANES");
    if (value == nullptr || value[0] == '\0')
      return 8;
    return std::max(1, std::min(32, std::atoi(value)));
  }();
  return lanes;
}

void submit_hipblas_dgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                          hipblasOperation_t transb, hipStream_t stream, int m,
                          int n, int k, double alpha, const double *A, int lda,
                          const double *B, int ldb, double beta, double *C,
                          int ldc) {
  HICBLAS_CHECK(hipblasSetStream(handle, stream));
  HICBLAS_CHECK(hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda,
                             B, ldb, &beta, C, ldc));
}

void run_dgemm_graph_dag(
    hipblasOperation_t transa, hipblasOperation_t transb,
    const cache_key<double> &key, int m, const int *n, const int *k,
    double alpha, const double *A, int lda, const int64_t *offsetsA,
    const double *B, const int *ldb, const int64_t *offsetsB, double beta,
    double *C, int ldc, const int64_t *offsetsC, int batchCount,
    hipStream_t stream, void *growing_allocator, bool synchronize) {
  using gemm_type = hipblas_gemm_grouped<double>;
  const bool debug = graph_debug_enabled();
  const auto callStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  const uint64_t sequence = debug ? ++graph_debug_sequence : 0;

  std::vector<int> active;
  active.reserve(batchCount);
  for (int i = 0; i < batchCount; ++i)
    if (m != 0 && n[i] != 0 && k[i] != 0)
      active.push_back(i);
  const int lanes = std::min(dgemm_graph_lane_limit(),
                             static_cast<int>(active.size()));
  if (lanes <= 1) {
    run_group_graph(gemm_type(transa, transb), key, m, n, k, alpha, A, lda,
                    offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
                    batchCount, stream, growing_allocator, synchronize);
    return;
  }

  {
    std::lock_guard<std::mutex> registration_lock(
        growing_allocator_registration_mutex);
    growing_allocator_register_free_c(growing_allocator,
                                      free_gemm_graph_cache<gemm_type>);
  }

  auto &cacheState = get_graph_cache_state<gemm_type>();
  std::unique_lock<std::mutex> cacheLock(cacheState.mutex);
  auto &graphCache = cacheState.graph_cache;
  auto graph = graphCache.find(key);
  const bool cacheMiss = graph == graphCache.end();
  std::shared_ptr<graph_exec_entry> graphExec;
  if (!cacheMiss)
    graphExec = graph->second;

  if (cacheMiss) {
    const auto buildStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    std::stable_sort(active.begin(), active.end(), [&](int left, int right) {
      return static_cast<uint64_t>(n[left]) * k[left] >
             static_cast<uint64_t>(n[right]) * k[right];
    });
    std::vector<std::vector<int>> laneGroups(lanes);
    std::vector<uint64_t> laneWork(lanes, 0);
    for (int groupIndex : active) {
      const int lane = static_cast<int>(
          std::min_element(laneWork.begin(), laneWork.end()) -
          laneWork.begin());
      laneGroups[lane].push_back(groupIndex);
      laneWork[lane] += static_cast<uint64_t>(n[groupIndex]) * k[groupIndex];
    }
    const auto scheduleEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};

    int device;
    HIC_CHECK(hipGetDevice(&device));
    std::vector<hipStream_t> laneStreams(lanes);
    std::vector<std::unique_ptr<hipblas_handle_owner>> laneHandles;
    laneHandles.reserve(lanes);
    double streamCreateMs = 0.0;
    double handleCreateMs = 0.0;
    double warmupEnqueueMs = 0.0;
    double warmupSyncMs = 0.0;

    // Lane-handle warmup must not pass work already queued on the caller's
    // OpenACC stream. This synchronization only occurs while building a graph.
    HIC_CHECK(hipStreamSynchronize(stream));
    const auto callerSyncEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    for (int lane = 0; lane < lanes; ++lane) {
      const auto streamCreateStart =
          debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
      HIC_CHECK(hipStreamCreate(&laneStreams[lane]));
      const auto streamCreateEnd =
          debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
      laneHandles.emplace_back(std::make_unique<hipblas_handle_owner>(device));
      const auto handleCreateEnd =
          debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
      const int warmupIndex = laneGroups[lane].front();
      submit_hipblas_dgemm(
          laneHandles[lane]->get(), transa, transb, laneStreams[lane], m,
          n[warmupIndex], k[warmupIndex], alpha, A + offsetsA[warmupIndex], lda,
          B + offsetsB[warmupIndex], ldb[warmupIndex], beta,
          C + offsetsC[warmupIndex], ldc);
      const auto warmupEnqueueEnd =
          debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
      HIC_CHECK(hipStreamSynchronize(laneStreams[lane]));
      if (debug) {
        const auto warmupSyncEnd = graph_debug_clock::now();
        streamCreateMs +=
            graph_debug_ms(streamCreateStart, streamCreateEnd);
        handleCreateMs += graph_debug_ms(streamCreateEnd, handleCreateEnd);
        warmupEnqueueMs +=
            graph_debug_ms(handleCreateEnd, warmupEnqueueEnd);
        warmupSyncMs += graph_debug_ms(warmupEnqueueEnd, warmupSyncEnd);
      }
    }

    const auto eventCreateStart =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    hipEvent_t forkEvent;
    HIC_CHECK(hipEventCreateWithFlags(&forkEvent, hipEventDisableTiming));
    std::vector<hipEvent_t> completionEvents(lanes - 1);
    for (hipEvent_t &completionEvent : completionEvents)
      HIC_CHECK(
          hipEventCreateWithFlags(&completionEvent, hipEventDisableTiming));
    const auto eventCreateEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};

    const auto captureBeginStart = eventCreateEnd;
    HIC_CHECK(hipStreamBeginCapture(laneStreams[0],
                                    hipStreamCaptureModeGlobal));
    const auto captureBeginEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    HIC_CHECK(hipEventRecord(forkEvent, laneStreams[0]));
    for (int lane = 1; lane < lanes; ++lane)
      HIC_CHECK(hipStreamWaitEvent(laneStreams[lane], forkEvent, 0));

    for (int lane = 0; lane < lanes; ++lane) {
      for (int groupIndex : laneGroups[lane])
        submit_hipblas_dgemm(
            laneHandles[lane]->get(), transa, transb, laneStreams[lane], m,
            n[groupIndex], k[groupIndex], alpha, A + offsetsA[groupIndex], lda,
            B + offsetsB[groupIndex], ldb[groupIndex], beta,
            C + offsetsC[groupIndex], ldc);
    }
    for (int lane = 1; lane < lanes; ++lane) {
      HIC_CHECK(
          hipEventRecord(completionEvents[lane - 1], laneStreams[lane]));
      HIC_CHECK(hipStreamWaitEvent(laneStreams[0],
                                   completionEvents[lane - 1], 0));
    }
    const auto captureEnqueueEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};

    hipGraph_t graphRaw;
    HIC_CHECK(hipStreamEndCapture(laneStreams[0], &graphRaw));
    const auto captureEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    hic_graph_owner capturedGraph{graphRaw};

    const auto instantiateStart = captureEnd;
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, capturedGraph.get(), nullptr,
                                  nullptr, 0));
    const auto instantiateEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    const auto handleDestroyStart = instantiateEnd;
    laneHandles.clear();
    const auto handleDestroyEnd =
        debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
    HIC_CHECK(hipEventDestroy(forkEvent));
    for (hipEvent_t completionEvent : completionEvents)
      HIC_CHECK(hipEventDestroy(completionEvent));
    for (hipStream_t laneStream : laneStreams)
      HIC_CHECK(hipStreamDestroy(laneStream));
    capturedGraph.reset();

    graphExec = std::make_shared<graph_exec_entry>(instance);
    graphCache.emplace(key, graphExec);
    if (debug) {
      const auto buildEnd = graph_debug_clock::now();
      const auto [minimum, maximum] =
          std::minmax_element(laneWork.begin(), laneWork.end());
      std::fprintf(
          stderr,
          "EC_GRAPH_DEBUG event=build mode=dag rank=%s pid=%ld seq=%" PRIu64
          " precision=f64 resol=%d m=%d blas=%d batch=%d active=%zu lanes=%d "
          "lane_work_min=%" PRIu64 " lane_work_max=%" PRIu64
          " schedule_ms=%.3f caller_sync_ms=%.3f stream_create_ms=%.3f "
          "handle_create_ms=%.3f warmup_enqueue_ms=%.3f warmup_sync_ms=%.3f "
          "event_create_ms=%.3f capture_begin_ms=%.3f capture_enqueue_ms=%.3f "
          "capture_end_ms=%.3f instantiate_ms=%.3f handle_destroy_ms=%.3f "
          "cleanup_ms=%.3f "
          " build_total_ms=%.3f\n",
          graph_debug_rank(), static_cast<long>(getpid()), sequence,
          key.resol_id, m, key.blas_id, batchCount, active.size(), lanes,
          *minimum, *maximum, graph_debug_ms(buildStart, scheduleEnd),
          graph_debug_ms(scheduleEnd, callerSyncEnd), streamCreateMs,
          handleCreateMs, warmupEnqueueMs, warmupSyncMs,
          graph_debug_ms(eventCreateStart, eventCreateEnd),
          graph_debug_ms(captureBeginStart, captureBeginEnd),
          graph_debug_ms(captureBeginEnd, captureEnqueueEnd),
          graph_debug_ms(captureEnqueueEnd, captureEnd),
          graph_debug_ms(instantiateStart, instantiateEnd),
          graph_debug_ms(handleDestroyStart, handleDestroyEnd),
          graph_debug_ms(handleDestroyEnd, buildEnd),
          graph_debug_ms(buildStart, buildEnd));
      std::fflush(stderr);
    }
  }
  cacheLock.unlock();

  const auto launchStart =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  graphExec->launch(stream);
  const auto launchEnd =
      debug ? graph_debug_clock::now() : graph_debug_clock::time_point{};
  auto syncEnd = launchEnd;
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
  if (debug)
    syncEnd = graph_debug_clock::now();
  if (debug) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=return mode=dag rank=%s pid=%ld "
                 "seq=%" PRIu64 " precision=f64 resol=%d m=%d blas=%d "
                 "cache=%s lanes=%d launch_ms=%.3f sync_ms=%.3f "
                 "host_total_ms=%.3f\n",
                 graph_debug_rank(), static_cast<long>(getpid()), sequence,
                 key.resol_id, m, key.blas_id, cacheMiss ? "miss" : "hit",
                 lanes, graph_debug_ms(launchStart, launchEnd),
                 graph_debug_ms(launchEnd, syncEnd),
                 graph_debug_ms(callStart, syncEnd));
    std::fflush(stderr);
  }
}

#ifndef USE_CUTLASS

void hipblas_sgemm_wrapper_grouped(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, float alpha, const float *A, int lda, const int64_t *offsetsA,
    const float *B, const int *ldb, const int64_t *offsetsB, float beta,
    float *C, int ldc, const int64_t *offsetsC, int batchCount,
    hipStream_t stream, void *growing_allocator, bool synchronize = true) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;
  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

#ifdef USE_GRAPHS_GEMM
  if (graph_debug_force_group())
    run_group(hipblas_gemm_grouped<float>(op_t1, op_t2), resol_id, m, n, k,
              alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
              offsetsC, batchCount, stream, blas_id, synchronize);
  else {
    const auto key = make_cache_key(
        resol_id, blas_id, static_cast<int>(op_t1), static_cast<int>(op_t2), m,
        n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
        offsetsC, batchCount);
    run_group_graph(hipblas_gemm_grouped<float>(op_t1, op_t2), key, m, n, k,
                    alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
                    offsetsC, batchCount, stream, growing_allocator,
                    synchronize);
  }
#else
  run_group(hipblas_gemm_grouped<float>(op_t1, op_t2), resol_id, m, n, k, alpha,
            A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
            batchCount, stream, blas_id, synchronize);
#endif
}

#endif

void hipblas_dgemm_wrapper_grouped(int resol_id, int blas_id, char transa,
                                   char transb, int m, const int *n,
                                   const int *k, double alpha, const double *A,
                                   int lda, const int64_t *offsetsA,
                                   const double *B, const int *ldb,
                                   const int64_t *offsetsB, double beta,
                                   double *C, int ldc, const int64_t *offsetsC,
                                   int batchCount, hipStream_t stream,
                                   void *growing_allocator,
                                   bool synchronize = true) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;
  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  std::vector<int> aligned_n;
  std::vector<int> aligned_k;
  align_standard_legendre_shapes(blas_id, n, k, batchCount, aligned_n,
                                  aligned_k);

  diagnose_dgemm_quantization(resol_id, blas_id, m, n, k, lda, offsetsA, ldb,
                              offsetsB, ldc, offsetsC, batchCount);

#ifdef USE_CUTLASS
  if (cutlass_grouped_dp_enabled() &&
      cutlass_grouped_dp_profitable(m, n, k, batchCount)) {
    cutlass_dgemm_wrapper_grouped_true(
        resol_id, blas_id, op_t1, op_t2, m, n, k, alpha, A, lda, offsetsA, B,
        ldb, offsetsB, beta, C, ldc, offsetsC, batchCount, stream,
        growing_allocator, synchronize, cutlass_host_schedule_enabled());
    return;
  }
#endif

  if (dgemm_bucketed_enabled() && dgemm_bucketed_selected(blas_id) &&
      dgemm_bucketed_profitable(m, n, k, ldc, offsetsC, batchCount)) {
    const auto key = make_cache_key(
        resol_id, blas_id, static_cast<int>(op_t1), static_cast<int>(op_t2), m,
        n, k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
        offsetsC, batchCount);
    run_dgemm_bucketed(op_t1, op_t2, key, m, n, k, alpha, A, lda, offsetsA, B,
                       ldb, offsetsB, beta, C, ldc, offsetsC, batchCount,
                       stream, growing_allocator, synchronize);
    return;
  }

#ifdef USE_GRAPHS_GEMM
  const auto key = make_cache_key(
      resol_id, blas_id, static_cast<int>(op_t1), static_cast<int>(op_t2), m, n,
      k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
      batchCount);
  const bool forceGroup = graph_debug_force_group();
  if (forceGroup) {
    run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
              alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
              offsetsC, batchCount, stream, blas_id, synchronize);
  } else {
    int activeGroups = 0;
    if (beta == 0.0) {
      for (int i = 0; i < batchCount && activeGroups < 2; ++i)
        if (m != 0 && n[i] != 0 && k[i] != 0)
          ++activeGroups;
    }
    const bool dagCanWarmItself = activeGroups >= 2;
    if (dagCanWarmItself) {
      run_dgemm_graph_dag(op_t1, op_t2, key, m, n, k, alpha, A, lda,
                          offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
                          batchCount, stream, growing_allocator, synchronize);
      return;
    }

    {
      std::lock_guard<std::mutex> registration_lock(
          growing_allocator_registration_mutex);
      growing_allocator_register_free_c(
          growing_allocator,
          free_gemm_graph_cache<hipblas_gemm_grouped<double>>);
    }
    auto &cacheState =
        get_graph_cache_state<hipblas_gemm_grouped<double>>();
    std::unique_lock<std::mutex> cacheLock(cacheState.mutex);
    const bool firstCall = cacheState.warmup_cache.emplace(key, true).second;
    // Keep this plan locked until its cold HIPBLAS submission has completed.
    if (firstCall) {
      run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
                offsetsC, batchCount, stream, blas_id, synchronize);
    } else {
      cacheLock.unlock();
      run_dgemm_graph_dag(op_t1, op_t2, key, m, n, k, alpha, A, lda, offsetsA,
                          B, ldb, offsetsB, beta, C, ldc, offsetsC, batchCount,
                          stream, growing_allocator, synchronize);
    }
  }
#else
  run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
            alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
            batchCount, stream, blas_id, synchronize);
#endif
}

hipStream_t hip_stream_from_value(std::intptr_t stream) {
  return reinterpret_cast<hipStream_t>(stream);
}

} // namespace

extern "C" {
void hipblas_dgemm_wrapper(char transa, char transb, int m, int n, int k,
                           double alpha, const double *A, int lda, int tda,
                           const double *B, int ldb, int tdb, double beta,
                           double *C, int ldc, int tdc, int batchCount,
                           std::intptr_t stream, void *growing_allocator) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  hipblasHandle_t handle = get_hipblas_handle();
  HICBLAS_CHECK(hipblasSetStream(handle, hip_stream_from_value(stream)));

  HICBLAS_CHECK(hipblasDgemmStridedBatched(
      handle, op_t1, op_t2, m, n, k, &alpha, (const double *)A, lda, tda,
      (const double *)B, ldb, tdb, &beta, (double *)C, ldc, tdc, batchCount));
}

void hipblas_sgemm_wrapper(char transa, char transb, int m, int n, int k,
                           float alpha, const float *A, int lda, int tda,
                           const float *B, int ldb, int tdb, float beta,
                           float *C, int ldc, int tdc, int batchCount,
                           std::intptr_t stream, void *growing_allocator) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  hipblasHandle_t handle = get_hipblas_handle();
  HICBLAS_CHECK(hipblasSetStream(handle, hip_stream_from_value(stream)));
  HICBLAS_CHECK(hipblasSgemmStridedBatched(
      handle, op_t1, op_t2, m, n, k, &alpha, (const float *)A, lda, tda,
      (const float *)B, ldb, tdb, &beta, (float *)C, ldc, tdc, batchCount));
}

void hipblas_sgemm_wrapper_grouped(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, float alpha, const float *A, int lda, const int64_t *offsetsA,
    const float *B, const int *ldb, const int64_t *offsetsB, float beta,
    float *C, int ldc, const int64_t *offsetsC, int batchCount,
    std::intptr_t stream,
    void *growing_allocator) {
#ifdef USE_CUTLASS
  cutlass_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator);
#else
  hipblas_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator);
#endif
}

void hipblas_sgemm_wrapper_grouped_async(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, float alpha, const float *A, int lda, const int64_t *offsetsA,
    const float *B, const int *ldb, const int64_t *offsetsB, float beta,
    float *C, int ldc, const int64_t *offsetsC, int batchCount,
    std::intptr_t stream,
    void *growing_allocator) {
#ifdef USE_CUTLASS
  cutlass_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator);
#else
  hipblas_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator,
                                false);
#endif
}

void hipblas_dgemm_wrapper_grouped(int resol_id, int blas_id, char transa,
                                   char transb, int m, const int *n,
                                   const int *k, double alpha, const double *A,
                                   int lda, const int64_t *offsetsA,
                                   const double *B, const int *ldb,
                                   const int64_t *offsetsB, double beta,
                                   double *C, int ldc, const int64_t *offsetsC,
                                   int batchCount, std::intptr_t stream,
                                   void *growing_allocator) {
  hipblas_dgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator);
}

void hipblas_dgemm_wrapper_grouped_async(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, double alpha, const double *A, int lda,
    const int64_t *offsetsA, const double *B, const int *ldb,
    const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, std::intptr_t stream,
    void *growing_allocator) {
  hipblas_dgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                hip_stream_from_value(stream), growing_allocator,
                                graph_debug_force_sync_async());
}

void hipblas_dgemm_wrapper_grouped_ordered_async(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, double alpha, const double *A, int lda,
    const int64_t *offsetsA, const double *B, const int *ldb,
    const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, std::intptr_t stream,
    void *growing_allocator) {
  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;
  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;
#ifdef USE_CUTLASS
  if (cutlass_grouped_dp_enabled() && cutlass_ordered_dp_enabled() &&
      beta == 1.0 && cutlass_grouped_dp_profitable(m, n, k, batchCount)) {
    cutlass_dgemm_wrapper_grouped_ordered_true(
        resol_id, blas_id, op_t1, op_t2, m, n, k, alpha, A, lda, offsetsA, B,
        ldb, offsetsB, beta, C, ldc, offsetsC, batchCount,
        hip_stream_from_value(stream), growing_allocator, false);
    return;
  }
#endif
  run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
            alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
            batchCount, hip_stream_from_value(stream), blas_id, false);
}

void clean_gemm(int resol_id) {
  erase_from_caches<hipblas_gemm_grouped<float>>(resol_id);
  erase_from_caches<hipblas_gemm_grouped<double>>(resol_id);
  erase_dgemm_bucketed_cache(resol_id);
#ifdef USE_CUTLASS
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_3xtf32, CUBLAS_OP_T, CUBLAS_OP_T>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_3xtf32, CUBLAS_OP_N, CUBLAS_OP_T>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_3xtf32, CUBLAS_OP_T, CUBLAS_OP_N>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_3xtf32, CUBLAS_OP_N, CUBLAS_OP_N>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_fp32, CUBLAS_OP_T, CUBLAS_OP_T>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_fp32, CUBLAS_OP_N, CUBLAS_OP_T>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_fp32, CUBLAS_OP_T, CUBLAS_OP_N>>(resol_id);
  erase_from_caches<detail::cutlass_sgemm_grouped<
      detail::CutlassType::cutlass_fp32, CUBLAS_OP_N, CUBLAS_OP_N>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_entry<CUBLAS_OP_T, CUBLAS_OP_T>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_entry<CUBLAS_OP_N, CUBLAS_OP_T>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_entry<CUBLAS_OP_T, CUBLAS_OP_N>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_entry<CUBLAS_OP_N, CUBLAS_OP_N>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<detail::cutlass_dgemm_grouped_entry<
      CUBLAS_OP_T, CUBLAS_OP_T,
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<detail::cutlass_dgemm_grouped_entry<
      CUBLAS_OP_N, CUBLAS_OP_T,
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<detail::cutlass_dgemm_grouped_entry<
      CUBLAS_OP_T, CUBLAS_OP_N,
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<detail::cutlass_dgemm_grouped_entry<
      CUBLAS_OP_N, CUBLAS_OP_N,
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>>(resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_ordered_entry<CUBLAS_OP_T, CUBLAS_OP_T>>(
      resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_ordered_entry<CUBLAS_OP_N, CUBLAS_OP_T>>(
      resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_ordered_entry<CUBLAS_OP_T, CUBLAS_OP_N>>(
      resol_id);
  erase_cutlass_dgemm_grouped_cache<
      detail::cutlass_dgemm_grouped_ordered_entry<CUBLAS_OP_N, CUBLAS_OP_N>>(
      resol_id);
#endif
}
}

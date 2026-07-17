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

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unistd.h>

#include "hicblas.h"
#ifdef USE_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#endif

#include "growing_allocator.h"

bool hip_alreadyAllocated_sgemm = false;
bool hip_alreadyAllocated_sgemm_handle = false;

hipblasHandle_t handle_hip_sgemm;

namespace {
using graph_debug_clock = std::chrono::steady_clock;

std::atomic<uint64_t> graph_debug_sequence{0};

bool graph_debug_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ECTRANS_GEMM_GRAPH_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool graph_debug_force_group() {
  const char *value = std::getenv("ECTRANS_GEMM_GRAPH_MODE");
  return value != nullptr && std::strcmp(value, "group") == 0;
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

struct cache_key {
  int resol_id;
  int m;
  int blas_id;

  bool operator==(const cache_key &other) const {
    return resol_id == other.resol_id && m == other.m &&
           blas_id == other.blas_id;
  }
  cache_key(int resol_id_, int m_, int blas_id_)
      : resol_id(resol_id_), m(m_), blas_id(blas_id_) {}
};
} // namespace
template <> struct std::hash<cache_key> {
  std::size_t operator()(const cache_key &k) const {
    return k.blas_id * 1000000 + k.resol_id * 10000 + k.m;
  }
};

namespace {
template <typename Gemm> auto &get_graph_cache() {
  // we store at most one graph per "m" (# fields) and "blas id" and resolution
  static std::unordered_map<cache_key, std::shared_ptr<hipGraphExec_t>>
      graphCache;
  return graphCache;
}
template <typename Gemm> auto &get_ptr_cache() {
  using real_t = typename Gemm::real_type;
  static std::unordered_map<
      cache_key, std::tuple<const real_t *, const real_t *, const real_t *>>
      ptrCache;
  return ptrCache;
}
template <typename Gemm> auto &get_signature_cache() {
  static std::unordered_map<cache_key, uint64_t> signatureCache;
  return signatureCache;
}

template <typename Gemm> void free_gemm_graph_cache(float *, size_t) {
  if (graph_debug_enabled()) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=cache_clear reason=allocator rank=%s "
                 "pid=%ld precision=%s graph_entries=%zu ptr_entries=%zu "
                 "signature_entries=%zu\n",
                 graph_debug_rank(), static_cast<long>(getpid()),
                 std::is_same<typename Gemm::real_type, double>::value ? "f64"
                                                                       : "f32",
                 get_graph_cache<Gemm>().size(), get_ptr_cache<Gemm>().size(),
                 get_signature_cache<Gemm>().size());
    std::fflush(stderr);
  }
  get_graph_cache<Gemm>().clear();
  get_ptr_cache<Gemm>().clear();
  get_signature_cache<Gemm>().clear();
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
  if (graph_debug_enabled()) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=cache_clear reason=resolution rank=%s "
                 "pid=%ld precision=%s resol=%d graph_entries=%zu "
                 "ptr_entries=%zu signature_entries=%zu\n",
                 graph_debug_rank(), static_cast<long>(getpid()),
                 std::is_same<typename Gemm::real_type, double>::value ? "f64"
                                                                       : "f32",
                 resol_id, get_graph_cache<Gemm>().size(),
                 get_ptr_cache<Gemm>().size(),
                 get_signature_cache<Gemm>().size());
    std::fflush(stderr);
  }
  erase_resol_from_cache(get_graph_cache<Gemm>(), resol_id);
  erase_resol_from_cache(get_ptr_cache<Gemm>(), resol_id);
  erase_resol_from_cache(get_signature_cache<Gemm>(), resol_id);
}

// this version is using graphs and caches the graphs
template <typename Gemm, typename Real>
void run_group_graph(Gemm &&gemm, int resol_id, int m, const int *n,
                     const int *k, Real alpha, const Real *A, int lda,
                     const int64_t *offsetsA, const Real *B, const int *ldb,
                     const int64_t *offsetsB, Real beta, Real *C, int ldc,
                     const int64_t *offsetsC, int batchCount,
                     hipStream_t stream, int blas_id, void *growing_allocator,
                     bool synchronize = true) {
  const auto callStart = graph_debug_clock::now();
  const bool debug = graph_debug_enabled();
  const uint64_t sequence = debug ? ++graph_debug_sequence : 0;
  const uint64_t signature =
      debug ? graph_debug_signature(m, n, k, lda, offsetsA, ldb, offsetsB,
                                    ldc, offsetsC, batchCount)
            : 0;
  const int activeGroups =
      debug ? graph_debug_active_groups(m, n, k, batchCount) : 0;

  growing_allocator_register_free_c(growing_allocator,
                                    free_gemm_graph_cache<Gemm>);

  // we store at most one graph per "m" (# fields) and "blas id"
  auto &graphCache = get_graph_cache<Gemm>();

  // we also store A, B, and C and recreate the graph if they change
  auto &ptrCache = get_ptr_cache<Gemm>();
  auto &signatureCache = get_signature_cache<Gemm>();

  auto key = cache_key{resol_id, m, blas_id};

  auto ptrs = ptrCache.find(key);
  const bool pointerChange =
      ptrs != ptrCache.end() &&
      (std::get<0>(ptrs->second) != A || std::get<1>(ptrs->second) != B ||
       std::get<2>(ptrs->second) != C);
  if (pointerChange) {
    // the plan is cached, but the pointers are not correct. we remove and
    // delete the graph, but we keep the hipblas handles, if this happens more
    // often, we should cache this...
    std::cout
        << "WARNING GEMM: POINTER CHANGE - Graph recreation might be slow."
        << std::endl;
    std::cout << "We have an entry with key {m=" << m << ", blas_id=" << blas_id
              << ", resol_id=" << resol_id << "}\n";
    std::cout << "Pointers: " << std::get<0>(ptrs->second) << ", "
              << std::get<1>(ptrs->second) << ", " << std::get<2>(ptrs->second)
              << " vs. " << A << ", " << B << ", " << C << std::endl;
    graphCache.erase(key);
    ptrCache.erase(key);
    signatureCache.erase(key);
  }

  auto graph = graphCache.find(key);
  const bool cacheMiss = graph == graphCache.end();
  const auto cachedSignature = signatureCache.find(key);
  const bool signatureChange =
      debug && !cacheMiss && cachedSignature != signatureCache.end() &&
      cachedSignature->second != signature;
  if (debug) {
    std::fprintf(
        stderr,
        "EC_GRAPH_DEBUG event=call mode=graph rank=%s pid=%ld seq=%" PRIu64
        " precision=%s resol=%d m=%d blas=%d batch=%d active=%d "
        "signature=%016" PRIx64 " cache=%s pointer_change=%d "
        "signature_change=%d graph_entries=%zu stream=%p A=%p B=%p C=%p\n",
        graph_debug_rank(), static_cast<long>(getpid()), sequence,
        std::is_same<Real, double>::value ? "f64" : "f32", resol_id, m,
        blas_id, batchCount, activeGroups, signature,
        cacheMiss ? "miss" : "hit", pointerChange ? 1 : 0,
        signatureChange ? 1 : 0, graphCache.size(),
        reinterpret_cast<void *>(stream), static_cast<const void *>(A),
        static_cast<const void *>(B), static_cast<void *>(C));
    std::fflush(stderr);
  }

  if (graph == graphCache.end()) {
    // this graph does not exist yet
    const auto createStart = graph_debug_clock::now();
    hipStream_t captureStream;
    HIC_CHECK(hipStreamCreate(&captureStream));
    const auto createEnd = graph_debug_clock::now();
    std::size_t graphNodes = 0;
    auto beginStart = createEnd;
    auto beginEnd = createEnd;
    auto enqueueEnd = createEnd;
    auto captureEnd = createEnd;
    auto instantiateEnd = createEnd;

#ifdef USE_CUTLASS
    hipGraph_t new_graph;
    hipGraphCreate(&new_graph, 0);
    for (int i = 0; i < batchCount; ++i) {
      if (m == 0 || n[i] == 0 || k[i] == 0)
        continue;

      HIC_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
      gemm(captureStream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
           ldb[i], beta, C + offsetsC[i], ldc);
      hipGraph_t my_graph;
      HIC_CHECK(hipStreamEndCapture(captureStream, &my_graph));
      hipGraphNode_t my_node;
      HIC_CHECK(
          hipGraphAddChildGraphNode(&my_node, new_graph, nullptr, 0, my_graph));
    }
    HIC_CHECK(hipGraphGetNodes(new_graph, nullptr, &graphNodes));
    const auto instantiateStart = graph_debug_clock::now();
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, new_graph, NULL, NULL, 0));
    instantiateEnd = graph_debug_clock::now();
    HIC_CHECK(hipGraphDestroy(new_graph));
#else
    beginStart = graph_debug_clock::now();
    HIC_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
    beginEnd = graph_debug_clock::now();
    for (int i = 0; i < batchCount; ++i) {
      if (m == 0 || n[i] == 0 || k[i] == 0)
        continue;

      gemm(captureStream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
           ldb[i], beta, C + offsetsC[i], ldc);
    }
    enqueueEnd = graph_debug_clock::now();
    hipGraph_t my_graph;
    HIC_CHECK(hipStreamEndCapture(captureStream, &my_graph));
    captureEnd = graph_debug_clock::now();
    HIC_CHECK(hipGraphGetNodes(my_graph, nullptr, &graphNodes));
    const auto instantiateStart = graph_debug_clock::now();
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, my_graph, NULL, NULL, 0));
    instantiateEnd = graph_debug_clock::now();
#endif
    const auto destroyStart = graph_debug_clock::now();
    HIC_CHECK(hipStreamDestroy(captureStream));
    const auto destroyEnd = graph_debug_clock::now();

    graphCache.insert({key, std::shared_ptr<hipGraphExec_t>(
                                new hipGraphExec_t{instance}, [](auto ptr) {
                                  HIC_CHECK(hipGraphExecDestroy(*ptr));
                                  delete ptr;
                                })});
    ptrCache.insert({key, std::make_tuple(A, B, C)});
    if (debug)
      signatureCache.insert_or_assign(key, signature);
    if (debug) {
      std::fprintf(
          stderr,
          "EC_GRAPH_DEBUG event=build rank=%s pid=%ld seq=%" PRIu64
          " precision=%s resol=%d m=%d blas=%d batch=%d active=%d nodes=%zu "
          "stream_create_ms=%.3f begin_capture_ms=%.3f enqueue_ms=%.3f "
          "end_capture_ms=%.3f instantiate_ms=%.3f stream_destroy_ms=%.3f "
          "build_total_ms=%.3f\n",
          graph_debug_rank(), static_cast<long>(getpid()), sequence,
          std::is_same<Real, double>::value ? "f64" : "f32", resol_id, m,
          blas_id, batchCount, activeGroups, graphNodes,
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

  const auto launchStart = graph_debug_clock::now();
  HIC_CHECK(hipGraphLaunch(*graphCache.at(key), stream));
  const auto launchEnd = graph_debug_clock::now();
  auto syncEnd = launchEnd;
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
  syncEnd = graph_debug_clock::now();
  if (debug) {
    std::fprintf(stderr,
                 "EC_GRAPH_DEBUG event=return mode=graph rank=%s pid=%ld "
                 "seq=%" PRIu64 " precision=%s resol=%d m=%d blas=%d "
                 "cache=%s launch_ms=%.3f sync_ms=%.3f host_total_ms=%.3f\n",
                 graph_debug_rank(), static_cast<long>(getpid()), sequence,
                 std::is_same<Real, double>::value ? "f64" : "f32", resol_id,
                 m, blas_id, cacheMiss ? "miss" : "hit",
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
  const auto callStart = graph_debug_clock::now();
  const bool debug = graph_debug_enabled();
  const uint64_t sequence = debug ? ++graph_debug_sequence : 0;
  const uint64_t signature =
      debug ? graph_debug_signature(m, n, k, lda, offsetsA, ldb, offsetsB,
                                    ldc, offsetsC, batchCount)
            : 0;
  const int activeGroups =
      debug ? graph_debug_active_groups(m, n, k, batchCount) : 0;
  const auto enqueueStart = graph_debug_clock::now();
  for (int i = 0; i < batchCount; ++i) {
    if (m == 0 || n[i] == 0 || k[i] == 0)
      continue;
    gemm(stream, m, n[i], k[i], alpha, A + offsetsA[i], lda, B + offsetsB[i],
         ldb[i], beta, C + offsetsC[i], ldc);
  }
  const auto enqueueEnd = graph_debug_clock::now();
  if (synchronize)
    HIC_CHECK(hipStreamSynchronize(stream));
  const auto syncEnd = graph_debug_clock::now();
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

hipblasHandle_t get_hipblas_handle() {
  static hipblasHandle_t handle;
  if (!handle)
    HICBLAS_CHECK(hipblasCreate(&handle));
  return handle;
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
  else
    run_group_graph(hipblas_gemm_grouped<float>(op_t1, op_t2), resol_id, m, n,
                    k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
                    offsetsC, batchCount, stream, blas_id, growing_allocator,
                    synchronize);
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

#ifdef USE_GRAPHS_GEMM
  if (graph_debug_force_group())
    run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
              alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
              offsetsC, batchCount, stream, blas_id, synchronize);
  else
    run_group_graph(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n,
                    k, alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc,
                    offsetsC, batchCount, stream, blas_id, growing_allocator,
                    synchronize);
#else
  run_group(hipblas_gemm_grouped<double>(op_t1, op_t2), resol_id, m, n, k,
            alpha, A, lda, offsetsA, B, ldb, offsetsB, beta, C, ldc, offsetsC,
            batchCount, stream, blas_id, synchronize);
#endif
}

} // namespace

extern "C" {
void hipblas_dgemm_wrapper(char transa, char transb, int m, int n, int k,
                           double alpha, const double *A, int lda, int tda,
                           const double *B, int ldb, int tdb, double beta,
                           double *C, int ldc, int tdc, int batchCount,
                           size_t stream, void *growing_allocator) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  hipblasHandle_t handle = get_hipblas_handle();
  HICBLAS_CHECK(hipblasSetStream(handle, *(hipStream_t *)stream));

  HICBLAS_CHECK(hipblasDgemmStridedBatched(
      handle, op_t1, op_t2, m, n, k, &alpha, (const double *)A, lda, tda,
      (const double *)B, ldb, tdb, &beta, (double *)C, ldc, tdc, batchCount));
}

void hipblas_sgemm_wrapper(char transa, char transb, int m, int n, int k,
                           float alpha, const float *A, int lda, int tda,
                           const float *B, int ldb, int tdb, float beta,
                           float *C, int ldc, int tdc, int batchCount,
                           void *growing_allocator) {

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;
  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  if (!hip_alreadyAllocated_sgemm_handle) {
    HICBLAS_CHECK(hipblasCreate(&handle_hip_sgemm));
    hip_alreadyAllocated_sgemm_handle = true;
  }
  HICBLAS_CHECK(hipblasSgemmStridedBatched(
      handle_hip_sgemm, op_t1, op_t2, m, n, k, &alpha, (const float *)A, lda,
      tda, (const float *)B, ldb, tdb, &beta, (float *)C, ldc, tdc,
      batchCount));
}

void hipblas_sgemm_wrapper_grouped(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, float alpha, const float *A, int lda, const int64_t *offsetsA,
    const float *B, const int *ldb, const int64_t *offsetsB, float beta,
    float *C, int ldc, const int64_t *offsetsC, int batchCount, size_t stream,
    void *growing_allocator) {
#ifdef USE_CUTLASS
  cutlass_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator);
#else
  hipblas_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator);
#endif
}

void hipblas_sgemm_wrapper_grouped_async(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, float alpha, const float *A, int lda, const int64_t *offsetsA,
    const float *B, const int *ldb, const int64_t *offsetsB, float beta,
    float *C, int ldc, const int64_t *offsetsC, int batchCount, size_t stream,
    void *growing_allocator) {
#ifdef USE_CUTLASS
  cutlass_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator);
#else
  hipblas_sgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator,
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
                                   int batchCount, size_t stream,
                                   void *growing_allocator) {
  hipblas_dgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator);
}

void hipblas_dgemm_wrapper_grouped_async(
    int resol_id, int blas_id, char transa, char transb, int m, const int *n,
    const int *k, double alpha, const double *A, int lda,
    const int64_t *offsetsA, const double *B, const int *ldb,
    const int64_t *offsetsB, double beta, double *C, int ldc,
    const int64_t *offsetsC, int batchCount, size_t stream,
    void *growing_allocator) {
  hipblas_dgemm_wrapper_grouped(resol_id, blas_id, transa, transb, m, n, k,
                                alpha, A, lda, offsetsA, B, ldb, offsetsB, beta,
                                C, ldc, offsetsC, batchCount,
                                *(hipStream_t *)stream, growing_allocator,
                                false);
}

void clean_gemm(int resol_id) {
  erase_from_caches<hipblas_gemm_grouped<float>>(resol_id);
  erase_from_caches<hipblas_gemm_grouped<double>>(resol_id);
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
#endif
}
}

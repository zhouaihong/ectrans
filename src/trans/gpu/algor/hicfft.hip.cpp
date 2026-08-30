#include "hicfft.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "growing_allocator.h"
#include "hicgraph.h"

#define fftSafeCall(err) __fftSafeCall(err, __FILE__, __LINE__)

// __global__ void debug(int varId, int N, HIP_DATA_TYPE_COMPLEX *x) {
//     for (int i = 0; i < N; i++)
//     {
//         HIP_DATA_TYPE_COMPLEX a = x[i];
//         double b = (double)a.x;
//         double c = (double)a.y;
//         if (varId == 0) printf("GPU: input[%d]=(%2.4f,%2.4f)\n",i+1,b,c);
//         if (varId == 1) printf("GPU: output[%d]=(%2.4f,%2.4f)\n",i+1,b,c);
//     }
// }

// __global__ void debugFloat(int varId, int N, HIP_DATA_TYPE_REAL *x) {
//     for (int i = 0; i < N; i++)
//     {
//         double a = (double)x[i];
//         if (varId == 0) printf("GPU: input[%d]=%2.4f\n",i+1,a);
//         if (varId == 1) printf("GPU: output[%d]=%2.4f\n",i+1,a);
//     }
// }

namespace {
using steady_clock = std::chrono::steady_clock;

bool fft_warmup_debug() {
  const char *value = std::getenv("ECTRANS_GPU_WARMUP_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int fft_plan_threads() {
#if CUDAGPU
  const char *value = std::getenv("ECTRANS_FFT_PLAN_THREADS");
  if (value != nullptr) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 1)
      return static_cast<int>(std::min(parsed, 64L));
  }
#endif
  return 1;
}

bool fft_managed_workspace() {
#if CUDAGPU
  const char *value = std::getenv("ECTRANS_FFT_MANAGED_WORKSPACE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
#else
  return false;
#endif
}

bool fft_reuse_equivalent_plans() {
#if CUDAGPU
  const char *value = std::getenv("ECTRANS_FFT_REUSE_EQUIVALENT_PLANS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
#else
  return false;
#endif
}

bool fft_preplan_opposite() {
#if CUDAGPU
  const char *value = std::getenv("ECTRANS_FFT_PREPLAN_OPPOSITE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
#else
  return false;
#endif
}

std::mutex &fft_debug_mutex() {
  static std::mutex mutex;
  return mutex;
}

size_t align_workspace(size_t bytes) {
  constexpr size_t alignment = 256;
  return (bytes + alignment - 1) / alignment * alignment;
}

double elapsed_ms(steady_clock::time_point begin,
                  steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Double {
  using real = double;
  using cmplx = hipfftDoubleComplex;
};
struct Float {
  using real = float;
  using cmplx = hipfftComplex;
};

template <class Type, hipfftType Direction> class hicfft_plan {
  using real = typename Type::real;
  using cmplx = typename Type::cmplx;

public:
  void exec(real *data_real, cmplx *data_complex) const {
    real *data_real_l = &data_real[offset];
    cmplx *data_complex_l = &data_complex[offset / 2];
    if constexpr (Direction == HIPFFT_R2C)
      fftSafeCall(hipfftExecR2C(*handle_ptr, data_real_l, data_complex_l));
    else if constexpr (Direction == HIPFFT_C2R)
      fftSafeCall(hipfftExecC2R(*handle_ptr, data_complex_l, data_real_l));
    else if constexpr (Direction == HIPFFT_D2Z)
      fftSafeCall(hipfftExecD2Z(*handle_ptr, data_real_l, data_complex_l));
    else if constexpr (Direction == HIPFFT_Z2D)
      fftSafeCall(hipfftExecZ2D(*handle_ptr, data_complex_l, data_real_l));
  }
  void prepare(hipStream_t stream) {
    fftSafeCall(hipfftSetStream(*handle_ptr, stream));
#if CUDAGPU
    if (workspace)
      fftSafeCall(hipfftSetWorkArea(
          *handle_ptr, static_cast<char *>(workspace.get()) + workspace_offset));
#endif
  }
  hicfft_plan(std::shared_ptr<hipfftHandle> handle_, int64_t offset_,
              std::shared_ptr<void> workspace_ = {},
              size_t workspace_offset_ = 0)
      : workspace(std::move(workspace_)),
        handle_ptr(std::move(handle_)), offset(offset_),
        workspace_offset(workspace_offset_) {}

private:
  // External work memory must outlive the cuFFT handle that refers to it.
  std::shared_ptr<void> workspace;
  std::shared_ptr<hipfftHandle> handle_ptr;
  int64_t offset;
  size_t workspace_offset;
};

std::shared_ptr<hipfftHandle> own_fft_handle(hipfftHandle handle) {
  return std::shared_ptr<hipfftHandle>(new hipfftHandle{handle}, [](auto ptr) {
    fftSafeCall(hipfftDestroy(*ptr));
    delete ptr;
  });
}

struct cache_key {
  int resol_id;
  int kfield;
  bool operator==(const cache_key &other) const {
    return resol_id == other.resol_id && kfield == other.kfield;
  }
  cache_key(int resol_id_, int kfield_)
      : resol_id(resol_id_), kfield(kfield_) {}
};
} // namespace

template <> struct std::hash<cache_key> {
  std::size_t operator()(const cache_key &k) const {
    return k.resol_id * 10000 + k.kfield;
  }
};

namespace {
// kfield -> handles
template <class Type, hipfftType Direction> auto &get_fft_plan_cache() {
  static std::unordered_map<cache_key,
                            std::vector<hicfft_plan<Type, Direction>>>
      fftPlansCache;
  return fftPlansCache;
}
// kfield -> graphs
template <class Type, hipfftType Direction> auto &get_graph_cache() {
  static std::unordered_map<cache_key, std::shared_ptr<hipGraphExec_t>>
      graphCache;
  return graphCache;
}
// kfield -> ptrs
template <class Type, hipfftType Direction> auto &get_ptr_cache() {
  using real = typename Type::real;
  using cmplx = typename Type::cmplx;
  static std::unordered_map<cache_key, std::pair<real *, cmplx *>> ptrCache;
  return ptrCache;
}

template <class Type, hipfftType Direction>
void free_fft_graph_cache(void *, size_t) {
  get_graph_cache<Type, Direction>().clear();
  get_ptr_cache<Type, Direction>().clear();
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
template <class Type, hipfftType Direction>
void erase_from_caches(int resol_id) {
  erase_resol_from_cache(get_fft_plan_cache<Type, Direction>(), resol_id);
  erase_resol_from_cache(get_graph_cache<Type, Direction>(), resol_id);
  erase_resol_from_cache(get_ptr_cache<Type, Direction>(), resol_id);
}

template <class Type, hipfftType Direction>
std::vector<hicfft_plan<Type, Direction>> plan_all(int resol_id, int kfield, int *loens,
                                                   int nfft, int64_t *offsets) {
  static constexpr bool is_forward =
      Direction == HIPFFT_R2C || Direction == HIPFFT_D2Z;

  auto key = cache_key{resol_id, kfield};
  auto &fftPlansCache = get_fft_plan_cache<Type, Direction>();
  auto fftPlans = fftPlansCache.find(key);
  if (fftPlans == fftPlansCache.end()) {
    // the fft plans do not exist yet
    std::vector<hicfft_plan<Type, Direction>> newPlans;
    newPlans.reserve(nfft);
    const bool managed_workspace = fft_managed_workspace();
    const bool reuse_equivalent_plans =
        managed_workspace && fft_reuse_equivalent_plans();
    std::vector<int> operation_to_plan(nfft);
    std::vector<int> plan_representatives;
    plan_representatives.reserve(nfft);
    std::map<std::pair<int, int>, int> equivalent_plans;
    for (int i = 0; i < nfft; ++i) {
      const int dist = offsets[i + 1] - offsets[i];
      if (reuse_equivalent_plans) {
        const auto signature = std::make_pair(loens[i], dist);
        auto inserted = equivalent_plans.emplace(
            signature, static_cast<int>(plan_representatives.size()));
        operation_to_plan[i] = inserted.first->second;
        if (inserted.second)
          plan_representatives.push_back(i);
      } else {
        operation_to_plan[i] = i;
        plan_representatives.push_back(i);
      }
    }

    const int unique_plans = static_cast<int>(plan_representatives.size());
    std::vector<hipfftHandle> handles(unique_plans);
    std::vector<size_t> plan_workspace_sizes(managed_workspace ? unique_plans
                                                               : 0);
    auto create_plan = [&](int plan_index) {
      const int i = plan_representatives[plan_index];
      int nloen = loens[i];

      int dist = offsets[i + 1] - offsets[i];
      int embed[] = {1};
#if CUDAGPU
      if (managed_workspace) {
        fftSafeCall(hipfftCreate(&handles[plan_index]));
        fftSafeCall(hipfftSetAutoAllocation(handles[plan_index], 0));
        fftSafeCall(hipfftMakePlanMany(
            handles[plan_index], 1, &nloen, embed, 1,
            is_forward ? dist : dist / 2,
            embed, 1, is_forward ? dist / 2 : dist, Direction, kfield,
            &plan_workspace_sizes[plan_index]));
        return;
      }
#endif
      fftSafeCall(hipfftPlanMany(
          &handles[plan_index], 1, &nloen, embed, 1,
          is_forward ? dist : dist / 2, embed, 1,
          is_forward ? dist / 2 : dist, Direction, kfield));
    };

    const auto plan_begin = steady_clock::now();
    const int plan_threads = std::min(fft_plan_threads(), unique_plans);
    if (plan_threads == 1) {
      for (int plan_index = 0; plan_index < unique_plans; ++plan_index)
        create_plan(plan_index);
    } else {
      int device = 0;
      HIC_CHECK(hipGetDevice(&device));
      std::atomic<int> next{0};
      std::vector<std::thread> workers;
      workers.reserve(plan_threads);
      for (int thread = 0; thread < plan_threads; ++thread) {
        workers.emplace_back([&, device]() {
          HIC_CHECK(hipSetDevice(device));
          for (int plan_index = next.fetch_add(1); plan_index < unique_plans;
               plan_index = next.fetch_add(1))
            create_plan(plan_index);
        });
      }
      for (auto &worker : workers)
        worker.join();
    }
    const auto plan_end = steady_clock::now();

    std::shared_ptr<void> workspace;
    std::vector<size_t> workspace_offsets(managed_workspace ? nfft : 0);
    size_t workspace_bytes = 0;
#if CUDAGPU
    if (managed_workspace) {
      for (int i = 0; i < nfft; ++i) {
        workspace_offsets[i] = workspace_bytes;
        workspace_bytes += align_workspace(
            plan_workspace_sizes[operation_to_plan[i]]);
      }
      if (workspace_bytes > 0) {
        void *workspace_raw = nullptr;
        HIC_CHECK(hipMalloc(&workspace_raw, workspace_bytes));
        workspace.reset(workspace_raw,
                        [](void *ptr) { HIC_CHECK(hipFree(ptr)); });
      }
    }
#endif
    const auto allocation_end = steady_clock::now();

    std::vector<std::shared_ptr<hipfftHandle>> handle_owners;
    handle_owners.reserve(unique_plans);
    for (auto handle : handles)
      handle_owners.push_back(own_fft_handle(handle));
    for (int i = 0; i < nfft; ++i) {
      newPlans.emplace_back(handle_owners[operation_to_plan[i]],
                            kfield * offsets[i], workspace,
                            managed_workspace ? workspace_offsets[i] : 0);
    }
    if (fft_warmup_debug()) {
      std::lock_guard<std::mutex> lock(fft_debug_mutex());
      std::cout << "EC_WARMUP_DEBUG event=fft_plan direction="
                << static_cast<int>(Direction) << " resol=" << resol_id
                << " kfield=" << kfield << " nfft=" << nfft
                << " unique_plans=" << unique_plans
                << " threads=" << plan_threads
                << " managed_workspace=" << managed_workspace
                << " reuse_equivalent_plans=" << reuse_equivalent_plans
                << " make_ms=" << elapsed_ms(plan_begin, plan_end)
                << " allocation_ms=" << elapsed_ms(plan_end, allocation_end)
                << " workspace_mib="
                << static_cast<double>(workspace_bytes) / (1024.0 * 1024.0)
                << " ms=" << elapsed_ms(plan_begin, allocation_end)
                << std::endl;
    }
    fftPlansCache.insert({key, newPlans});
  }
  return fftPlansCache.find(key)->second;
}

template <class Type, hipfftType Direction>
void run_group_graph(typename Type::real *data_real,
                     typename Type::cmplx *data_complex, int resol_id, int kfield, int *loens,
                     int64_t *offsets, int nfft, void *growing_allocator) {

  growing_allocator_register_free_c(growing_allocator,
                                    free_fft_graph_cache<Type, Direction>);

  // if the pointers are changed, we need to update the graph
  auto &ptrCache = get_ptr_cache<Type, Direction>();     // kfield -> ptrs
  auto &graphCache = get_graph_cache<Type, Direction>(); // kfield -> graphs

  auto key = cache_key{resol_id, kfield};
  auto ptrs = ptrCache.find(key);
  if (ptrs != ptrCache.end() && (ptrs->second.first != data_real ||
                                 ptrs->second.second != data_complex)) {
    // the plan is cached, but the pointers are not correct. we remove and
    // delete the graph, but we keep the FFT plans, if this happens more often,
    // we should cache this...
    std::cout << "WARNING FFT: POINTER CHANGE --> THIS MIGHT BE SLOW"
              << std::endl;
    graphCache.erase(key);
    ptrCache.erase(key);
  }

  auto graph = graphCache.find(key);
  if (graph == graphCache.end()) {
    // this graph does not exist yet
    const auto build_begin = steady_clock::now();
    auto plans =
        plan_all<Type, Direction>(resol_id, kfield, loens, nfft, offsets);
    const auto plans_ready = steady_clock::now();

    // create a temporary stream
    hipStream_t stream;
    HIC_CHECK(hipStreamCreate(&stream));

#if HIPGPU
    for (auto &plan : plans)
      plan.prepare(stream);
    // now create the graph
    HIC_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (auto &plan : plans) {
      plan.exec(data_real, data_complex);
    }
    hipGraph_t my_graph_raw;
    HIC_CHECK(hipStreamEndCapture(stream, &my_graph_raw));
    hic_graph_owner my_graph{my_graph_raw};
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, my_graph.get(), NULL, NULL, 0));
    my_graph.reset();
#endif
#if CUDAGPU
    // now create the graph
    hipGraph_t new_graph_raw;
    HIC_CHECK(hipGraphCreate(&new_graph_raw, 0));
    hic_graph_owner new_graph{new_graph_raw};
    std::vector<hic_graph_owner> child_graphs;
    child_graphs.reserve(plans.size());
    for (auto &plan : plans) {
      plan.prepare(stream);
      HIC_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
      plan.exec(data_real, data_complex);
      hipGraph_t my_graph_raw;
      HIC_CHECK(hipStreamEndCapture(stream, &my_graph_raw));
      hic_graph_owner my_graph{my_graph_raw};
      hipGraphNode_t my_node;
      HIC_CHECK(
          hipGraphAddChildGraphNode(&my_node, new_graph.get(), nullptr, 0,
                                    my_graph.get()));
      child_graphs.push_back(std::move(my_graph));
    }
    const auto capture_done = steady_clock::now();
    hipGraphExec_t instance;
    HIC_CHECK(hipGraphInstantiate(&instance, new_graph.get(), NULL, NULL, 0));
    const auto instantiate_done = steady_clock::now();
    child_graphs.clear();
    new_graph.reset();
#endif
    HIC_CHECK(hipStreamDestroy(stream));

    graphCache.insert({key, std::shared_ptr<hipGraphExec_t>(
                                new hipGraphExec_t{instance}, [](auto ptr) {
                                  HIC_CHECK(hipGraphExecDestroy(*ptr));
                                  delete ptr;
                                })});
    ptrCache.insert({key, std::make_pair(data_real, data_complex)});
    if (fft_warmup_debug()) {
      std::lock_guard<std::mutex> lock(fft_debug_mutex());
      std::cout << "EC_WARMUP_DEBUG event=fft_graph direction="
                << static_cast<int>(Direction) << " resol=" << resol_id
                << " kfield=" << kfield << " nfft=" << nfft
                << " plan_ms=" << elapsed_ms(build_begin, plans_ready)
#if CUDAGPU
                << " capture_ms=" << elapsed_ms(plans_ready, capture_done)
                << " instantiate_ms="
                << elapsed_ms(capture_done, instantiate_done)
#endif
                << " total_ms=" << elapsed_ms(build_begin, steady_clock::now())
                << std::endl;
    }
  }

  /* running in stream 0 */
  HIC_CHECK(hipGraphLaunch(*graphCache.at(key), 0));
  HIC_CHECK(hipStreamSynchronize(0));
}

template <class Type, hipfftType Direction>
void run_group(typename Type::real *data_real,
               typename Type::cmplx *data_complex, int resol_id, int kfield, int *loens,
               int64_t *offsets, int nfft, void *growing_allocator) {
  auto plans = plan_all<Type, Direction>(resol_id, kfield, loens, nfft, offsets);

  for (auto &plan : plans) {
    plan.prepare(0);
    plan.exec(data_real, data_complex);
  }
  HIC_CHECK(hipDeviceSynchronize());
}

template <hipfftType Direction> constexpr hipfftType opposite_direction() {
  if constexpr (Direction == HIPFFT_R2C)
    return HIPFFT_C2R;
  else if constexpr (Direction == HIPFFT_C2R)
    return HIPFFT_R2C;
  else if constexpr (Direction == HIPFFT_D2Z)
    return HIPFFT_Z2D;
  else
    return HIPFFT_D2Z;
}

template <class Type, hipfftType Direction>
void run_with_opposite_preplan(typename Type::real *data_real,
                               typename Type::cmplx *data_complex,
                               int resol_id, int kfield, int *loens,
                               int64_t *offsets, int nfft,
                               void *growing_allocator) {
  static constexpr hipfftType Opposite = opposite_direction<Direction>();
  const auto key = cache_key{resol_id, kfield};
  auto &opposite_cache = get_fft_plan_cache<Type, Opposite>();
  const bool preplan = fft_preplan_opposite() &&
                       opposite_cache.find(key) == opposite_cache.end();
  const auto begin = steady_clock::now();
  std::thread preplan_thread;
  if (preplan) {
    int device = 0;
    HIC_CHECK(hipGetDevice(&device));
    preplan_thread = std::thread([=]() {
      HIC_CHECK(hipSetDevice(device));
      plan_all<Type, Opposite>(resol_id, kfield, loens, nfft, offsets);
    });

    plan_all<Type, Direction>(resol_id, kfield, loens, nfft, offsets);
    preplan_thread.join();
  }

  // cuFFT plan creation in another thread invalidates global stream capture.
#ifdef USE_GRAPHS_FFT
  run_group_graph<Type, Direction>(data_real, data_complex, resol_id, kfield,
                                   loens, offsets, nfft, growing_allocator);
#else
  run_group<Type, Direction>(data_real, data_complex, resol_id, kfield, loens,
                             offsets, nfft, growing_allocator);
#endif

  if (preplan && fft_warmup_debug()) {
    std::lock_guard<std::mutex> lock(fft_debug_mutex());
    std::cout << "EC_WARMUP_DEBUG event=fft_preplan direction="
              << static_cast<int>(Direction)
              << " opposite=" << static_cast<int>(Opposite)
              << " resol=" << resol_id << " kfield=" << kfield
              << " nfft=" << nfft
              << " total_ms=" << elapsed_ms(begin, steady_clock::now())
              << std::endl;
  }
}
} // namespace

extern "C" {
void execute_dir_fft_float(float *data_real, hipfftComplex *data_complex,
                           int resol_id, int kfield, int *loens, int64_t *offsets, int nfft,
                           void *growing_allocator) {
  run_with_opposite_preplan<Float, HIPFFT_R2C>(
      data_real, data_complex, resol_id, kfield, loens, offsets, nfft,
      growing_allocator);
}
void execute_inv_fft_float(hipfftComplex *data_complex, float *data_real,
                           int resol_id, int kfield, int *loens, int64_t *offsets, int nfft,
                           void *growing_allocator) {
  run_with_opposite_preplan<Float, HIPFFT_C2R>(
      data_real, data_complex, resol_id, kfield, loens, offsets, nfft,
      growing_allocator);
}
void execute_dir_fft_double(double *data_real,
                            hipfftDoubleComplex *data_complex, int resol_id, int kfield,
                            int *loens, int64_t *offsets, int nfft,
                            void *growing_allocator) {
  run_with_opposite_preplan<Double, HIPFFT_D2Z>(
      data_real, data_complex, resol_id, kfield, loens, offsets, nfft,
      growing_allocator);
}
void execute_inv_fft_double(hipfftDoubleComplex *data_complex,
                            double *data_real, int resol_id, int kfield, int *loens,
                            int64_t *offsets, int nfft, void *growing_allocator) {
  run_with_opposite_preplan<Double, HIPFFT_Z2D>(
      data_real, data_complex, resol_id, kfield, loens, offsets, nfft,
      growing_allocator);
}

void clean_fft(int resol_id) {
  erase_from_caches<Float, HIPFFT_R2C>(resol_id);
  erase_from_caches<Float, HIPFFT_C2R>(resol_id);
  erase_from_caches<Double, HIPFFT_D2Z>(resol_id);
  erase_from_caches<Double, HIPFFT_Z2D>(resol_id);
}
}

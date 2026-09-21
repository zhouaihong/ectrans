#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call) do { int s = int(call); if (s) { \
  std::fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #call, s); \
  throw std::runtime_error(#call); } } while (0)

template<class T> std::vector<T> read_array(std::istream &in, size_t n) {
  std::vector<T> v(n);
  if (!in.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)))
    throw std::runtime_error("truncated fixture");
  return v;
}
template<class T> struct Buffer {
  T *ptr = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
  explicit Buffer(const std::vector<T> &v) : Buffer(v.size()) {
    CHECK(cudaMemcpy(ptr, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { cudaFree(ptr); }
  std::vector<T> host() const {
    std::vector<T> v(size);
    CHECK(cudaMemcpy(v.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

struct Header { int32_t magic, m, degree, rows, rhs, samples, width; };
struct HostPart {
  int cols;
  std::vector<double> x, p, a, g, w, reverse_w;
  std::vector<int32_t> reverse_offsets, reverse_indices;
};
struct Fixture {
  Header h{};
  HostPart part[2];
  std::vector<int32_t> indices;
  explicit Fixture(const char *path) {
    std::ifstream in(path, std::ios::binary);
    static_assert(sizeof(Header) == 28, "fixture header size");
    if (!in.read(reinterpret_cast<char *>(&h), sizeof(h)) || h.magic != 0x52534d31 ||
        h.degree != 1280 || h.m < 0 || h.m > 1279 || h.rows < 1 || h.rows > 1280 ||
        h.samples < 1 || h.samples > 1280 || h.rhs != 242 || h.width < 2 || h.width > 512)
      throw std::runtime_error("invalid resample header");
    for (int parity = 0; parity < 2; ++parity) {
      auto &p = part[parity];
      p.cols = (h.degree - h.m - parity) / 2 + 1;
      p.x = read_array<double>(in, size_t(h.rhs) * p.cols);
      p.p = read_array<double>(in, size_t(h.rows) * p.cols);
      p.a = read_array<double>(in, size_t(h.samples) * p.cols);
      p.g = read_array<double>(in, size_t(h.rhs) * h.rows);
    }
    indices = read_array<int32_t>(in, size_t(h.rows) * h.width);
    for (int k : indices) if (k < 0 || k >= h.samples) throw std::runtime_error("forward index");
    for (auto &p : part) p.w = read_array<double>(in, indices.size());
    for (auto &p : part) {
      int n = read_array<int32_t>(in, 1)[0];
      if (n < 1 || n > h.rows * h.width) throw std::runtime_error("reverse nnz");
      p.reverse_offsets = read_array<int32_t>(in, h.samples + 1);
      p.reverse_indices = read_array<int32_t>(in, n);
      p.reverse_w = read_array<double>(in, n);
      if (p.reverse_offsets.front() != 0 || p.reverse_offsets.back() != n ||
          !std::is_sorted(p.reverse_offsets.begin(), p.reverse_offsets.end()))
        throw std::runtime_error("reverse offsets");
      for (int k : p.reverse_indices) if (k < 0 || k >= h.rows) throw std::runtime_error("reverse index");
      for (auto *v : {&p.x, &p.p, &p.a, &p.g, &p.w, &p.reverse_w})
        for (double value : *v) if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture");
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture bytes");
  }
};

struct Part {
  int cols;
  Buffer<double> x, p, a, g, w, reverse_w, sampled, out, dense, gathered, back, dense_back;
  Buffer<int32_t> offsets, reverse_indices;
  Part(const Header &h, const HostPart &p)
      : cols(p.cols), x(p.x), p(p.p), a(p.a), g(p.g), w(p.w), reverse_w(p.reverse_w),
        sampled(size_t(h.samples) * h.rhs), out(size_t(h.rows) * h.rhs),
        dense(size_t(h.rows) * h.rhs), gathered(size_t(h.samples) * h.rhs),
        back(size_t(h.rhs) * cols), dense_back(size_t(h.rhs) * cols),
        offsets(p.reverse_offsets), reverse_indices(p.reverse_indices) {}
};

// Field-leading access: a warp shares interpolation indices/weights and reads adjacent RHS.
__global__ void interpolate(const double *se, const double *so, double *ye, double *yo,
                             const int32_t *indices, const double *we, const double *wo,
                             int rhs, int rows, int width) {
  int field = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (field >= rhs || row >= rows) return;
  double even = 0., odd = 0.;
  for (int j = 0; j < width; ++j) {
    int q = row * width + j, k = indices[q] * rhs + field;
    even = fma(we[q], se[k], even);
    odd = fma(wo[q], so[k], odd);
  }
  ye[row * rhs + field] = even;
  yo[row * rhs + field] = odd;
}

// Exact transpose of the stored sparse map; deterministic gathers, no atomics.
__global__ void gather_transpose(const double *g, double *out, const int32_t *offsets,
                                 const int32_t *indices, const double *weights,
                                 int rhs, int rows) {
  int field = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (field >= rhs || row >= rows) return;
  double value = 0.;
  for (int q = offsets[row]; q < offsets[row + 1]; ++q)
    value = fma(weights[q], g[indices[q] * rhs + field], value);
  out[row * rhs + field] = value;
}

struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  template<class F> Graph(cudaStream_t stream, F op) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    op();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  }
  ~Graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
};
double timing(cudaGraphExec_t graph, cudaStream_t stream) {
  cudaEvent_t begin, end;
  CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  CHECK(cudaEventRecord(begin, stream));
  for (int i = 0; i < 100; ++i) CHECK(cudaGraphLaunch(graph, stream));
  CHECK(cudaEventRecord(end, stream)); CHECK(cudaEventSynchronize(end));
  float ms;
  CHECK(cudaEventElapsedTime(&ms, begin, end));
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
  return ms / 100.;
}

struct Error { double l2 = 0., linf = 0.; bool finite = true; };
Error error(const std::vector<double> &a, const std::vector<double> &b, int rhs) {
  if (a.size() != b.size() || a.size() % rhs) throw std::runtime_error("error shapes");
  Error out;
  for (int field = 0; field < rhs; ++field) {
    long double norm = 0., diff = 0.;
    double peak = 0., delta = 0.;
    for (size_t j = field; j < a.size(); j += rhs) {
      out.finite = out.finite && std::isfinite(a[j]) && std::isfinite(b[j]);
      long double d = static_cast<long double>(a[j]) - b[j];
      diff += d * d; norm += static_cast<long double>(b[j]) * b[j];
      peak = std::max(peak, std::abs(b[j])); delta = std::max(delta, std::abs(double(d)));
    }
    if (norm == 0 || peak == 0) throw std::runtime_error("zero reference column");
    out.l2 = std::max(out.l2, std::sqrt(double(diff / norm)));
    out.linf = std::max(out.linf, delta / peak);
  }
  return out;
}
std::vector<double> hemispheres(const std::vector<double> &e, const std::vector<double> &o) {
  std::vector<double> result(2 * e.size());
  for (size_t i = 0; i < e.size(); ++i) {
    result[i] = e[i] + o[i]; result[i + e.size()] = e[i] - o[i];
  }
  return result;
}
bool valid(const Error &e) { return e.finite && e.l2 <= 1e-13 && e.linf <= 1e-13; }

bool run(const Fixture &f, const char *label) {
  const auto &h = f.h;
  cudaStream_t stream;
  cublasHandle_t blas;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas, stream));
  CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  CHECK(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
  bool passed;
  {
    Part even(h, f.part[0]), odd(h, f.part[1]);
    Buffer<int32_t> indices(f.indices);
    const double one = 1., zero = 0.;
    auto dense_forward = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, h.rows, p->cols,
          &one, p->x.ptr, h.rhs, p->p.ptr, h.rows, &zero, p->dense.ptr, h.rhs));
    };
    auto compressed = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, h.samples, p->cols,
          &one, p->x.ptr, h.rhs, p->a.ptr, h.samples, &zero, p->sampled.ptr, h.rhs));
    };
    auto interpolate_forward = [&] {
      interpolate<<<dim3((h.rhs + 127) / 128, h.rows), 128, 0, stream>>>(
        even.sampled.ptr, odd.sampled.ptr, even.out.ptr, odd.out.ptr,
        indices.ptr, even.w.ptr, odd.w.ptr, h.rhs, h.rows, h.width);
      CHECK(cudaGetLastError());
    };
    auto candidate_forward = [&] { compressed(); interpolate_forward(); };
    auto dense_reverse = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rhs, p->cols, h.rows,
          &one, p->g.ptr, h.rhs, p->p.ptr, h.rows, &zero, p->dense_back.ptr, h.rhs));
    };
    auto gather = [&] {
      for (Part *p : {&even, &odd})
        gather_transpose<<<dim3((h.rhs + 127) / 128, h.samples), 128, 0, stream>>>(
          p->g.ptr, p->gathered.ptr, p->offsets.ptr, p->reverse_indices.ptr,
          p->reverse_w.ptr, h.rhs, h.samples);
      CHECK(cudaGetLastError());
    };
    auto reverse_gemm = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rhs, p->cols, h.samples,
          &one, p->gathered.ptr, h.rhs, p->a.ptr, h.samples, &zero, p->back.ptr, h.rhs));
    };
    auto candidate_reverse = [&] { gather(); reverse_gemm(); };
    for (int i = 0; i < 3; ++i) {
      dense_forward(); candidate_forward(); dense_reverse(); candidate_reverse();
    }
    CHECK(cudaStreamSynchronize(stream));
    auto forward = error(hemispheres(even.out.host(), odd.out.host()),
                         hemispheres(even.dense.host(), odd.dense.host()), h.rhs);
    auto be = even.back.host(), bo = odd.back.host();
    auto de = even.dense_back.host(), d_o = odd.dense_back.host();
    be.insert(be.end(), bo.begin(), bo.end()); de.insert(de.end(), d_o.begin(), d_o.end());
    auto reverse = error(be, de, h.rhs);
    passed = valid(forward) && valid(reverse);
    size_t free, total;
    CHECK(cudaMemGetInfo(&free, &total));
    std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"m\":%d,\"rows\":%d,"
      "\"samples\":%d,\"width\":%d,\"forward_l2\":%.17g,\"forward_linf\":%.17g,"
      "\"adjoint_l2\":%.17g,\"adjoint_linf\":%.17g,\"pass\":%s,\"device_used_mib\":%.1f}\n",
      label, h.m, h.rows, h.samples, h.width, forward.l2, forward.linf,
      reverse.l2, reverse.linf, passed ? "true" : "false", double(total - free) / 1048576.);
    if (passed) {
      Graph df(stream, dense_forward), cf(stream, candidate_forward);
      Graph dr(stream, dense_reverse), cr(stream, candidate_reverse);
      Graph cg(stream, compressed), ip(stream, interpolate_forward);
      Graph rg(stream, reverse_gemm), ga(stream, gather);
      for (int repeat = 0; repeat < 8; ++repeat) {
        for (int direction = 0; direction < 2; ++direction) {
          auto a = direction ? dr.exec : df.exec, b = direction ? cr.exec : cf.exec;
          double dense_ms, candidate_ms;
          if (repeat % 2) { candidate_ms = timing(b, stream); dense_ms = timing(a, stream); }
          else { dense_ms = timing(a, stream); candidate_ms = timing(b, stream); }
          std::printf("{\"kind\":\"operator_diagnostic\",\"case\":\"%s\",\"direction\":\"%s\","
            "\"repeat\":%d,\"dense_ms\":%.9g,\"resample_ms\":%.9g,\"ratio\":%.9g}\n",
            label, direction ? "adjoint" : "forward", repeat + 1, dense_ms, candidate_ms,
            candidate_ms / dense_ms);
        }
      }
      std::printf("{\"kind\":\"stages\",\"case\":\"%s\",\"sample_gemm_ms\":%.9g,"
        "\"interpolation_ms\":%.9g,\"transpose_gather_ms\":%.9g,\"transpose_gemm_ms\":%.9g}\n",
        label, timing(cg.exec, stream), timing(ip.exec, stream), timing(ga.exec, stream), timing(rg.exec, stream));
    }
    std::fflush(stdout);
  }
  CHECK(cublasDestroy(blas)); CHECK(cudaStreamDestroy(stream));
  return passed;
}

int main(int argc, char **argv) {
  try {
    bool inspect = argc > 1 && std::string(argv[1]) == "--inspect";
    int first = inspect ? 2 : 1;
    if (argc <= first) throw std::runtime_error("fixture files required");
    if (!inspect) {
      cudaDeviceProp device;
      CHECK(cudaGetDeviceProperties(&device, 0));
      std::printf("device=%s runtime=%d scope=independent operator\n", device.name, CUDART_VERSION);
    }
    bool passed = true;
    for (int i = first; i < argc; ++i) {
      Fixture f(argv[i]);
      if (inspect) std::printf("fixture=%s m=%d rows=%d samples=%d width=%d ok\n",
        argv[i], f.h.m, f.h.rows, f.h.samples, f.h.width);
      else {
        std::string label(argv[i]); label = label.substr(label.find_last_of('/') + 1);
        passed = run(f, label.c_str()) && passed;
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "resample_probe: %s\n", e.what());
    return 2;
  }
}

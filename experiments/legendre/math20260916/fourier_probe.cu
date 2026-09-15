#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call) do { int status = int(call); if (status != 0) { \
  std::fprintf(stderr, "%s:%d %s failed: %d\n", __FILE__, __LINE__, #call, status); \
  throw std::runtime_error(#call); } } while (0)

template <class T> struct Device {
  T *data = nullptr;
  size_t size;
  explicit Device(size_t count) : size(count) { CHECK(cudaMalloc(&data, size * sizeof(T))); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  ~Device() { cudaFree(data); }
  void read(std::istream &stream, int transpose_rows = 0) {
    std::vector<T> host(size);
    if (!stream.read(reinterpret_cast<char *>(host.data()), size * sizeof(T)))
      throw std::runtime_error("truncated fixture");
    if (transpose_rows) {
      const size_t cols = size / transpose_rows;
      std::vector<T> transposed(size);
      for (int i = 0; i < transpose_rows; ++i)
        for (size_t j = 0; j < cols; ++j) transposed[i + j * transpose_rows] = host[i * cols + j];
      host.swap(transposed);
    }
    CHECK(cudaMemcpy(data, host.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> host() const {
    std::vector<T> result(size);
    CHECK(cudaMemcpy(result.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
};

struct Header {
  int32_t order, degree, rows, rhs, fft, width;
};
static Header read_header(std::istream &stream) {
  Header h{};
  static_assert(sizeof(Header) == 24, "fixture header");
  if (!stream.read(reinterpret_cast<char *>(&h), sizeof(h)) ||
      h.order < 0 || h.order > 1279 || h.degree != 1280 ||
      h.rows < 1 || h.rows > 1280 || h.rhs != 242 || h.fft != 5120 || h.width != 40)
    throw std::runtime_error("invalid fixture header");
  return h;
}

static int first_frequency(const Header &h, int parity) {
  int first = (h.order + parity) % 2;
  if ((h.order % 2) && first == 0) first = 2;
  return first;
}

struct Part {
  int cols, first, frequencies, prefix;
  Device<double> connection, input, polynomial, converted, dense, candidate;
  Part(std::istream &stream, const Header &h, int parity)
      : cols((h.degree - h.order - parity) / 2 + 1),
        first(first_frequency(h, parity)), frequencies((h.degree - first) / 2 + 1),
        prefix((h.order + parity - first) / 2),
        connection(size_t(frequencies) * cols), input(size_t(h.rhs) * cols),
        polynomial(size_t(h.rows) * cols), converted(size_t(h.rhs) * frequencies),
        dense(size_t(h.rows) * h.rhs), candidate(size_t(h.rows) * h.rhs) {
    connection.read(stream);
    input.read(stream);
    polynomial.read(stream);
  }
};

__global__ void pack_spectrum(const double *even, const double *odd,
                              const cufftDoubleComplex *twiddle, cufftDoubleComplex *spectrum,
                              int rhs, int half, int degree, int order) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rhs * half) return;
  const int field = index % rhs, k = index / rhs;
  cufftDoubleComplex value{0., 0.};
  if (k <= degree && (!(order % 2) || k != 0)) {
    const int parity = (k + order) % 2;
    int first = (order + parity) % 2;
    if ((order % 2) && first == 0) first = 2;
    const int row = (k - first) / 2;
    const double a = (parity ? odd : even)[field + row * rhs] * (k == 0 ? 1. : .5);
    value = (order % 2) ? cufftDoubleComplex{a * twiddle[k].y, -a * twiddle[k].x}
                       : cufftDoubleComplex{a * twiddle[k].x, a * twiddle[k].y};
  }
  spectrum[size_t(field) * half + k] = value;
}

__global__ void interpolate_fold(const double *sampled, const int32_t *indices,
                                 const double *weights, double *even, double *odd,
                                 int rows, int rhs, int fft, int width) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread >= rows * rhs) return;
  const int field = thread / rows, latitude = thread % rows;
  const int index = field + latitude * rhs;
  const double *values = sampled + size_t(field) * fft;
  const int north = latitude, south = 2 * rows - 1 - latitude;
  double a = 0., b = 0.;
  for (int j = 0; j < width; ++j) {
    a += weights[north + j * 2 * rows] * values[indices[north + j * 2 * rows]];
    b += weights[south + j * 2 * rows] * values[indices[south + j * 2 * rows]];
  }
  even[index] = .5 * (a + b);
  odd[index] = .5 * (a - b);
}

struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  template <class Operation> Graph(cudaStream_t stream, Operation operation) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    operation();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  }
  ~Graph() {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
};

static double time_graph(cudaGraphExec_t graph, cudaStream_t stream) {
  cudaEvent_t begin, end;
  CHECK(cudaEventCreate(&begin));
  CHECK(cudaEventCreate(&end));
  CHECK(cudaEventRecord(begin, stream));
  constexpr int iterations = 100;
  for (int i = 0; i < iterations; ++i) CHECK(cudaGraphLaunch(graph, stream));
  CHECK(cudaEventRecord(end, stream));
  CHECK(cudaEventSynchronize(end));
  float elapsed;
  CHECK(cudaEventElapsedTime(&elapsed, begin, end));
  CHECK(cudaEventDestroy(begin));
  CHECK(cudaEventDestroy(end));
  return elapsed / iterations;
}

static bool run_case(const char *path) {
  std::ifstream file(path, std::ios::binary);
  Header h = read_header(file);
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas;
  CHECK(cublasCreate(&blas));
  CHECK(cublasSetStream(blas, stream));
  CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  CHECK(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
  cufftHandle fft;
  CHECK(cufftPlan1d(&fft, h.fft, CUFFT_Z2D, h.rhs));
  CHECK(cufftSetStream(fft, stream));
  bool passed;
  {
    Part even(file, h, 0), odd(file, h, 1);
    Device<int32_t> indices(size_t(2) * h.rows * h.width);
    Device<double> weights(indices.size);
    indices.read(file, 2 * h.rows);
    weights.read(file, 2 * h.rows);
    if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture data");
    for (auto index : indices.host())
      if (index < 0 || index >= h.fft) throw std::runtime_error("invalid interpolation index");
    const int half = h.fft / 2 + 1;
    Device<cufftDoubleComplex> spectrum(size_t(h.rhs) * half), twiddle(half);
    Device<double> sampled(size_t(h.rhs) * h.fft);
    std::vector<cufftDoubleComplex> phase(half);
    for (int k = 0; k < half; ++k) {
      const double angle = std::acos(-1.) * k / (2. * h.fft);
      phase[k] = {std::cos(angle), std::sin(angle)};
    }
    CHECK(cudaMemcpy(twiddle.data, phase.data(), phase.size() * sizeof(phase[0]), cudaMemcpyHostToDevice));
    const double one = 1., zero = 0.;
    auto baseline = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, h.rows, p->cols,
                          &one, p->input.data, h.rhs, p->polynomial.data, h.rows,
                          &zero, p->dense.data, h.rhs));
    };
    auto candidate = [&] {
      for (Part *p : {&even, &odd}) {
        if (p->prefix)
          CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, p->prefix, p->cols,
                            &one, p->input.data, h.rhs, p->connection.data, p->frequencies,
                            &zero, p->converted.data, h.rhs));
        CHECK(cublasDtrmm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, h.rhs, p->cols, &one,
                         p->connection.data + p->prefix, p->frequencies, p->input.data,
                         h.rhs, p->converted.data + size_t(p->prefix) * h.rhs, h.rhs));
      }
      pack_spectrum<<<(h.rhs * half + 255) / 256, 256, 0, stream>>>(
          even.converted.data, odd.converted.data, twiddle.data, spectrum.data,
          h.rhs, half, h.degree, h.order);
      CHECK(cudaGetLastError());
      CHECK(cufftExecZ2D(fft, spectrum.data, sampled.data));
      interpolate_fold<<<(h.rows * h.rhs + 255) / 256, 256, 0, stream>>>(
          sampled.data, indices.data, weights.data, even.candidate.data, odd.candidate.data,
          h.rows, h.rhs, h.fft, h.width);
      CHECK(cudaGetLastError());
    };
    for (int warm = 0; warm < 3; ++warm) { baseline(); candidate(); }
    CHECK(cudaStreamSynchronize(stream));
    auto de = even.dense.host(), da = odd.dense.host();
    auto ce = even.candidate.host(), ca = odd.candidate.host();
    double max_l2 = 0., max_linf = 0.;
    bool finite = true;
    for (int field = 0; field < h.rhs; ++field) {
      long double norm = 0., error = 0.;
      double peak = 0., difference = 0.;
      for (int row = 0; row < h.rows; ++row) {
        const size_t j = field + size_t(row) * h.rhs;
        for (int sign : {-1, 1}) {
          double reference = de[j] + sign * da[j], actual = ce[j] + sign * ca[j];
          finite = finite && std::isfinite(reference) && std::isfinite(actual);
          const long double delta = static_cast<long double>(actual) - reference;
          norm += static_cast<long double>(reference) * reference;
          error += delta * delta;
          peak = std::max(peak, std::abs(reference));
          difference = std::max(difference, std::abs(double(delta)));
        }
      }
      if (norm == 0. || peak == 0.) throw std::runtime_error("zero reference column");
      max_l2 = std::max(max_l2, std::sqrt(double(error / norm)));
      max_linf = std::max(max_linf, difference / peak);
    }
    passed = finite && max_l2 <= 1e-13 && max_linf <= 1e-13;
    size_t free_bytes, total_bytes;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("{\"kind\":\"validation\",\"m\":%d,\"rhs\":%d,\"relative_l2\":%.17g,"
                "\"normalized_linf\":%.17g,\"pass\":%s,\"device_used_mib\":%.1f}\n",
                h.order, h.rhs, max_l2, max_linf, passed ? "true" : "false",
                double(total_bytes - free_bytes) / (1024 * 1024));
    if (passed) {
      Graph dense_graph(stream, baseline), fft_graph(stream, candidate);
      for (int repeat = 0; repeat < 8; ++repeat) {
        double a, b;
        if (repeat % 2 == 0) {
          a = time_graph(dense_graph.exec, stream); b = time_graph(fft_graph.exec, stream);
        } else {
          b = time_graph(fft_graph.exec, stream); a = time_graph(dense_graph.exec, stream);
        }
        std::printf("{\"kind\":\"operator_diagnostic\",\"m\":%d,\"repeat\":%d,"
                    "\"dense_ms\":%.9g,\"fourier_ms\":%.9g,\"ratio\":%.9g}\n",
                    h.order, repeat + 1, a, b, b / a);
      }
    }
    std::fflush(stdout);
  }
  CHECK(cufftDestroy(fft));
  CHECK(cublasDestroy(blas));
  CHECK(cudaStreamDestroy(stream));
  return passed;
}

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("fixture paths required");
    if (std::string(argv[1]) == "--inspect") {
      for (int i = 2; i < argc; ++i) {
        std::ifstream stream(argv[i], std::ios::binary);
        Header h = read_header(stream);
        size_t expected = sizeof(Header) + size_t(2) * h.rows * h.width * (sizeof(int32_t) + sizeof(double));
        for (int parity = 0; parity < 2; ++parity) {
          int cols = (h.degree - h.order - parity) / 2 + 1;
          int frequencies = (h.degree - first_frequency(h, parity)) / 2 + 1;
          expected += size_t(cols) * (frequencies + h.rhs + h.rows) * sizeof(double);
        }
        stream.seekg(0, std::ios::end);
        if (size_t(stream.tellg()) != expected) throw std::runtime_error("fixture size mismatch");
        std::printf("fixture=%s m=%d bytes=%zu ok\n", argv[i], h.order, expected);
      }
      return 0;
    }
    cudaDeviceProp device;
    CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("device=%s cuda_runtime=%d\n", device.name, CUDART_VERSION);
    bool passed = true;
    for (int i = 1; i < argc; ++i) passed = run_case(argv[i]) && passed;
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "fourier_probe: %s\n", error.what());
    return 2;
  }
}

// Native construction of the same normalized associated Legendre matrices.
// Recurrence: DLMF 14.10; __float128 routines: GCC libquadmath documentation.
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <quadmath.h>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> T root(T x) { return std::sqrt(x); }
template<> __float128 root(__float128 x) { return sqrtq(x); }

template<class T> T integer_power(T x, int n) {
  T answer = 1;
  for (; n; n >>= 1) {
    if (n & 1) answer *= x;
    if (n > 1) x *= x;
  }
  return answer;
}

struct Matrix {
  int m, degree, rows;
  std::vector<double> points, reference;
  explicit Matrix(const char *path) {
    std::ifstream in(path, std::ios::binary);
    int32_t h[4];
    if (!in.read(reinterpret_cast<char *>(h), sizeof(h)) || h[0] != 0x4c4d5231 ||
        h[1] < 0 || h[2] < h[1] || h[2] > 2560 || h[3] < 1 || h[3] > 8192)
      throw std::runtime_error("invalid matrix header");
    m = h[1]; degree = h[2]; rows = h[3];
    points.resize(rows);
    reference.resize(size_t(rows) * (degree - m + 1));
    if (!in.read(reinterpret_cast<char *>(points.data()), points.size() * sizeof(double)) ||
        !in.read(reinterpret_cast<char *>(reference.data()), reference.size() * sizeof(double)))
      throw std::runtime_error("truncated matrix");
    char byte;
    if (in.read(&byte, 1)) throw std::runtime_error("trailing matrix bytes");
    for (double x : points)
      if (!std::isfinite(x) || std::abs(x) > 1.) throw std::runtime_error("invalid node");
    for (double x : reference)
      if (!std::isfinite(x)) throw std::runtime_error("nonfinite reference");
  }
};

template<class T> std::vector<double> construct(const Matrix &v) {
  T diagonal = 2 * v.m + 1;
  for (int k = 1; k <= v.m; ++k) diagonal *= T(2 * k - 1) / T(2 * k);
  diagonal = root(diagonal);
  std::vector<T> eps(v.degree + 1);
  for (int n = v.m + 1; n <= v.degree; ++n)
    eps[n] = root(T(n * n - v.m * v.m) / T(4 * n * n - 1));
  const T first_factor = root(T(2 * v.m + 3));
  std::vector<double> out(v.reference.size());
  for (int row = 0; row < v.rows; ++row) {
    const T x = T(v.points[row]);
    T previous = diagonal * integer_power(root(T(1) - x * x), v.m);
    out[row] = double(previous);
    if (v.degree == v.m) continue;
    T current = first_factor * x * previous;
    out[size_t(v.rows) + row] = double(current);
    for (int n = v.m + 2; n <= v.degree; ++n) {
      const T next = (x * current - eps[n - 1] * previous) / eps[n];
      previous = current;
      current = next;
      out[size_t(n - v.m) * v.rows + row] = double(current);
    }
  }
  return out;
}

template<class T> bool measure(const Matrix &v, const std::string &label, const char *method) {
  std::vector<double> out;
  double seconds = 0;
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    out = construct<T>(v);
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  }
  double error = 0;
  size_t different = 0;
  bool finite = true;
  for (int col = 0; col <= v.degree - v.m; ++col) {
    double peak = 0, delta = 0;
    for (int row = 0; row < v.rows; ++row) {
      const size_t i = size_t(col) * v.rows + row;
      finite = finite && std::isfinite(out[i]);
      peak = std::max(peak, std::abs(v.reference[i]));
      delta = std::max(delta, std::abs(out[i] - v.reference[i]));
      different += out[i] != v.reference[i];
    }
    if (peak == 0 && delta != 0) finite = false;
    if (peak != 0) error = std::max(error, delta / peak);
  }
  const bool pass = finite && error < 3e-14;
  std::printf("{\"kind\":\"construction\",\"case\":\"%s\",\"method\":\"%s\","
      "\"m\":%d,\"degree\":%d,\"rows\":%d,\"mean_seconds\":%.9g,"
      "\"basis_max_error\":%.17g,\"different_values\":%zu,\"total_values\":%zu,"
      "\"pass\":%s}\n", label.c_str(), method, v.m, v.degree, v.rows, seconds / 3,
      error, different, out.size(), pass ? "true" : "false");
  return pass;
}

int main(int argc, char **argv) {
  try {
    const bool inspect = argc > 1 && std::string(argv[1]) == "--inspect";
    const int first = inspect ? 2 : 1;
    if (argc <= first) throw std::runtime_error("matrix inputs required");
    std::printf("construction_precision double_bits=%d long_double_bits=%d quad_bits=%d\n",
                DBL_MANT_DIG, LDBL_MANT_DIG, FLT128_MANT_DIG);
    bool passed = true;
    for (int i = first; i < argc; ++i) {
      Matrix v(argv[i]);
      std::string label(argv[i]);
      label = label.substr(label.find_last_of('/') + 1);
      if (inspect) {
        std::printf("matrix=%s m=%d degree=%d rows=%d input=valid\n",
                    label.c_str(), v.m, v.degree, v.rows);
        continue;
      }
      measure<double>(v, label, "double");
      measure<long double>(v, label, "long_double");
      passed = measure<__float128>(v, label, "quad") && passed;
      std::fflush(stdout);
    }
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "construct_probe: %s\n", e.what());
    return 2;
  }
}

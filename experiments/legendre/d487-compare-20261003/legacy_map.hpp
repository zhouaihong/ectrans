// Reuse the immutable fixture reader, errors and dense reference from 252050.
#define main untiled_probe_main
#include "resample_probe.cu"
#undef main
#include <memory>

struct HostTiles {
  int input_rows, output_rows, tile, span = 0, groups, padded_rows;
  std::vector<int> starts;
  std::vector<double> weights;
  size_t nnz = 0;
  HostTiles(const Fixture &f, int width, bool transpose)
      : input_rows(transpose ? f.h.rows : f.h.samples),
        output_rows(transpose ? f.h.samples : f.h.rows), tile(width),
        groups((output_rows + tile - 1) / tile), padded_rows(groups * tile) {
    auto visit = [&](int parity, int row, auto emit) {
      const auto &p = f.part[parity];
      if (transpose) {
        for (int q = p.reverse_offsets[row]; q < p.reverse_offsets[row + 1]; ++q)
          emit(p.reverse_indices[q], p.reverse_w[q]);
      } else {
        for (int q = row * f.h.width; q < (row + 1) * f.h.width; ++q)
          emit(f.indices[q], p.w[q]);
      }
    };
    starts.resize(2 * groups);
    for (int parity = 0; parity < 2; ++parity) {
      for (int group = 0; group < groups; ++group) {
        int lo = input_rows, hi = -1;
        for (int row = group * tile; row < std::min(output_rows, (group + 1) * tile); ++row)
          visit(parity, row, [&](int col, double) { lo = std::min(lo, col); hi = std::max(hi, col); ++nnz; });
        if (lo > hi) { starts[parity * groups + group] = 0; continue; }
        starts[parity * groups + group] = lo;
        span = std::max(span, hi - lo + 1);
      }
    }
    span = std::min(input_rows, (span + 7) / 8 * 8);
    weights.assign(size_t(2) * groups * span * tile, 0.);
    for (int parity = 0; parity < 2; ++parity) {
      for (int group = 0; group < groups; ++group) {
        const int batch = parity * groups + group;
        int &start = starts[batch];
        start = std::min(start, input_rows - span);
        for (int j = 0; j < tile && group * tile + j < output_rows; ++j) {
          visit(parity, group * tile + j, [&](int col, double value) {
            if (col < start || col >= start + span) throw std::runtime_error("tile range");
            weights[size_t(batch) * span * tile + j * span + col - start] += value;
          });
        }
      }
    }
    // Verify the packing with independent row accumulation, including duplicate boundary indices.
    for (int parity = 0; parity < 2; ++parity) {
      for (int row = 0; row < output_rows; ++row) {
        std::vector<double> exact(input_rows, 0.);
        visit(parity, row, [&](int col, double value) { exact[col] += value; });
        int batch = parity * groups + row / tile;
        for (int col = 0; col < input_rows; ++col) {
          int k = col - starts[batch];
          double actual = k >= 0 && k < span ?
            weights[size_t(batch) * span * tile + (row % tile) * span + k] : 0.;
          if (actual != exact[col]) throw std::runtime_error("tile packing changed the sparse map");
        }
      }
    }
  }
};

struct BatchedMap {
  const HostTiles &h;
  int rhs;
  Buffer<double> weights, even, odd;
  std::unique_ptr<Buffer<const double *>> input_ptrs, weight_ptrs;
  std::unique_ptr<Buffer<double *>> output_ptrs;
  BatchedMap(const HostTiles &host, int fields, const double *a, const double *b)
      : h(host), rhs(fields), weights(h.weights), even(size_t(rhs) * h.padded_rows),
        odd(size_t(rhs) * h.padded_rows) {
    std::vector<const double *> inputs, matrices;
    std::vector<double *> outputs;
    for (int parity = 0; parity < 2; ++parity) {
      for (int g = 0; g < h.groups; ++g) {
        int batch = parity * h.groups + g;
        inputs.push_back((parity ? b : a) + size_t(rhs) * h.starts[batch]);
        matrices.push_back(weights.ptr + size_t(batch) * h.span * h.tile);
        outputs.push_back((parity ? odd.ptr : even.ptr) + size_t(g) * h.tile * rhs);
      }
    }
    input_ptrs = std::make_unique<Buffer<const double *>>(inputs);
    weight_ptrs = std::make_unique<Buffer<const double *>>(matrices);
    output_ptrs = std::make_unique<Buffer<double *>>(outputs);
  }
  void apply(cublasHandle_t blas) {
    const double one = 1., zero = 0.;
    CHECK(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, rhs, h.tile, h.span,
      &one, input_ptrs->ptr, rhs, weight_ptrs->ptr, h.span, &zero,
      output_ptrs->ptr, rhs, 2 * h.groups));
  }
  std::vector<double> output(int parity) const {
    auto values = (parity ? odd : even).host();
    for (size_t i = size_t(rhs) * h.output_rows; i < values.size(); ++i)
      if (values[i] != 0.) throw std::runtime_error("nonzero padded output");
    values.resize(size_t(rhs) * h.output_rows);
    return values;
  }
};


std::vector<double> padded_grid(const Header &h, const HostPart &v) {
  auto values = v.g;
  // Only the allocation tail is zero: all real input columns retain their common addresses.
  values.resize(size_t(h.rhs) * ((h.rows + 31) / 32 * 32), 0.);
  return values;
}

struct DensePart {
  int cols;
  Buffer<double> x, p, g, out, back;
  DensePart(const Header &h, const HostPart &v)
      : cols(v.cols), x(v.x), p(v.p), g(padded_grid(h, v)),
        out(size_t(h.rhs) * h.rows), back(size_t(h.rhs) * cols) {}
};
struct DenseReference {
  std::unique_ptr<DensePart> part[2];
  explicit DenseReference(const Fixture &f) {
    for (int p = 0; p < 2; ++p) part[p] = std::make_unique<DensePart>(f.h, f.part[p]);
  }
};

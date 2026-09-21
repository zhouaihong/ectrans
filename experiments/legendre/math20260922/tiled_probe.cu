// Reuse the immutable fixture reader, errors and dense reference from 252050.
#define main untiled_probe_main
#include "../math20260921/resample_probe.cu"
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
        if (lo > hi) throw std::runtime_error("empty local interpolation block");
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

bool run_tiled(const Fixture &f, int tile, const char *label) {
  const auto &h = f.h;
  HostTiles forward_plan(f, tile, false), reverse_plan(f, tile, true);
  cudaStream_t stream;
  cublasHandle_t blas;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas, stream));
  CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  CHECK(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
  bool passed;
  {
    Part even(h, f.part[0]), odd(h, f.part[1]);
    BatchedMap forward_map(forward_plan, h.rhs, even.sampled.ptr, odd.sampled.ptr);
    BatchedMap reverse_map(reverse_plan, h.rhs, even.g.ptr, odd.g.ptr);
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
    auto interpolate_forward = [&] { forward_map.apply(blas); };
    auto candidate_forward = [&] { compressed(); interpolate_forward(); };
    auto dense_reverse = [&] {
      for (Part *p : {&even, &odd})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rhs, p->cols, h.rows,
          &one, p->g.ptr, h.rhs, p->p.ptr, h.rows, &zero, p->dense_back.ptr, h.rhs));
    };
    auto gather = [&] { reverse_map.apply(blas); };
    auto reverse_gemm = [&] {
      for (int parity = 0; parity < 2; ++parity) {
        Part *p = parity ? &odd : &even;
        const double *g = parity ? reverse_map.odd.ptr : reverse_map.even.ptr;
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rhs, p->cols, h.samples,
          &one, g, h.rhs, p->a.ptr, h.samples, &zero, p->back.ptr, h.rhs));
      }
    };
    auto candidate_reverse = [&] { gather(); reverse_gemm(); };
    for (int i = 0; i < 3; ++i) {
      dense_forward(); candidate_forward(); dense_reverse(); candidate_reverse();
    }
    CHECK(cudaStreamSynchronize(stream));
    Error forward = error(hemispheres(forward_map.output(0), forward_map.output(1)),
                          hemispheres(even.dense.host(), odd.dense.host()), h.rhs);
    reverse_map.output(0); reverse_map.output(1);
    auto actual = even.back.host(), odd_actual = odd.back.host();
    auto reference = even.dense_back.host(), odd_reference = odd.dense_back.host();
    actual.insert(actual.end(), odd_actual.begin(), odd_actual.end());
    reference.insert(reference.end(), odd_reference.begin(), odd_reference.end());
    Error reverse = error(actual, reference, h.rhs);
    passed = valid(forward) && valid(reverse);
    const double dense_cost = 2. * h.rhs * h.rows * (even.cols + odd.cols);
    const double sample_cost = 2. * h.rhs * h.samples * (even.cols + odd.cols);
    auto cost = [&](const HostTiles &p) { return sample_cost + 4. * h.rhs * p.groups * p.tile * p.span; };
    size_t free, total;
    CHECK(cudaMemGetInfo(&free, &total));
    std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"m\":%d,\"tile\":%d,"
      "\"forward_l2\":%.17g,\"forward_linf\":%.17g,\"adjoint_l2\":%.17g,"
      "\"adjoint_linf\":%.17g,\"forward_flops_ratio\":%.9g,\"adjoint_flops_ratio\":%.9g,"
      "\"forward_span\":%d,\"adjoint_span\":%d,\"pass\":%s,\"device_used_mib\":%.1f}\n",
      label, h.m, tile, forward.l2, forward.linf, reverse.l2, reverse.linf,
      cost(forward_plan) / dense_cost, cost(reverse_plan) / dense_cost,
      forward_plan.span, reverse_plan.span, passed ? "true" : "false", double(total - free) / 1048576.);
    if (passed) {
      Graph df(stream, dense_forward), cf(stream, candidate_forward);
      Graph dr(stream, dense_reverse), cr(stream, candidate_reverse);
      Graph cg(stream, compressed), ip(stream, interpolate_forward), ga(stream, gather), rg(stream, reverse_gemm);
      for (int repeat = 0; repeat < 8; ++repeat) {
        for (int direction = 0; direction < 2; ++direction) {
          auto a = direction ? dr.exec : df.exec, b = direction ? cr.exec : cf.exec;
          double dense_ms, candidate_ms;
          if (repeat % 2) { candidate_ms = timing(b, stream); dense_ms = timing(a, stream); }
          else { dense_ms = timing(a, stream); candidate_ms = timing(b, stream); }
          std::printf("{\"kind\":\"operator_diagnostic\",\"case\":\"%s\",\"direction\":\"%s\","
            "\"repeat\":%d,\"dense_ms\":%.9g,\"resample_ms\":%.9g,\"ratio\":%.9g}\n",
            label, direction ? "adjoint" : "forward", repeat + 1, dense_ms, candidate_ms, candidate_ms / dense_ms);
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
    bool passed = true;
    for (int i = first; i < argc; ++i) {
      Fixture f(argv[i]);
      for (int tile : {16, 32, 64}) {
        std::string label(argv[i]); label = label.substr(label.find_last_of('/') + 1) + "-tile" + std::to_string(tile);
        if (inspect) {
          HostTiles forward(f, tile, false), reverse(f, tile, true);
          std::printf("fixture=%s tile=%d forward_span=%d reverse_span=%d packing=exact ok\n",
            argv[i], tile, forward.span, reverse.span);
        } else passed = run_tiled(f, tile, label.c_str()) && passed;
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "tiled_probe: %s\n", e.what());
    return 2;
  }
}

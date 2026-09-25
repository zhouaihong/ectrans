// Reuse the immutable fixture reader, errors and dense reference from 252050.
#define main untiled_probe_main
#include "../../math20260921/resample_probe.cu"
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


struct HostOverlap {
  int samples;
  std::vector<int> offsets, indices;
  explicit HostOverlap(const HostTiles &h) : samples(h.input_rows) {
    for (int p = 0; p < 2; ++p) {
      offsets.push_back(int(indices.size()));
      for (int row = 0; row < samples; ++row) {
        for (int g = 0; g < h.groups; ++g) {
          int batch = p * h.groups + g, local = row - h.starts[batch];
          if (local >= 0 && local < h.span) indices.push_back(batch * h.span + local);
        }
        offsets.push_back(int(indices.size()));
      }
    }
    if (indices.size() != size_t(2) * h.groups * h.span)
      throw std::runtime_error("overlap map lost a partial output");
  }
};

__global__ void sum_overlap(const double *partials, double *even, double *odd,
                            const int *offsets, const int *indices, int rhs, int samples) {
  int field = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y, p = blockIdx.z;
  if (field >= rhs || row >= samples) return;
  int r = p * (samples + 1) + row;
  double sum = 0.;
  for (int q = offsets[r]; q < offsets[r + 1]; ++q)
    sum += partials[size_t(indices[q]) * rhs + field];
  (p ? odd : even)[size_t(row) * rhs + field] = sum;
}

struct BlockTranspose {
  const BatchedMap &forward;
  HostOverlap host;
  Buffer<double> partials, even, odd;
  Buffer<int> offsets, indices;
  std::unique_ptr<Buffer<const double *>> input_ptrs;
  std::unique_ptr<Buffer<double *>> output_ptrs;
  BlockTranspose(const BatchedMap &map, const double *g0, const double *g1)
      : forward(map), host(map.h), partials(size_t(2) * map.h.groups * map.h.span * map.rhs),
        even(size_t(host.samples) * map.rhs), odd(size_t(host.samples) * map.rhs),
        offsets(host.offsets), indices(host.indices) {
    std::vector<const double *> inputs;
    std::vector<double *> outputs;
    for (int p = 0; p < 2; ++p) {
      for (int g = 0; g < map.h.groups; ++g) {
        inputs.push_back((p ? g1 : g0) + size_t(g) * map.h.tile * map.rhs);
        outputs.push_back(partials.ptr + size_t(p * map.h.groups + g) * map.h.span * map.rhs);
      }
    }
    input_ptrs = std::make_unique<Buffer<const double *>>(inputs);
    output_ptrs = std::make_unique<Buffer<double *>>(outputs);
  }
  void multiply(cublasHandle_t blas) {
    const double one = 1., zero = 0.;
    const auto &h = forward.h;
    CHECK(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, forward.rhs, h.span, h.tile,
      &one, input_ptrs->ptr, forward.rhs, forward.weight_ptrs->ptr, h.span,
      &zero, output_ptrs->ptr, forward.rhs, 2 * h.groups));
  }
  void reduce(cudaStream_t stream) {
    dim3 grid((forward.rhs + 127) / 128, host.samples, 2);
    sum_overlap<<<grid, 128, 0, stream>>>(partials.ptr, even.ptr, odd.ptr,
      offsets.ptr, indices.ptr, forward.rhs, host.samples);
    CHECK(cudaGetLastError());
  }
  void apply(cublasHandle_t blas, cudaStream_t stream) { multiply(blas); reduce(stream); }
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

bool run_tiled(const Fixture &f, DenseReference &reference, int tile, const char *label,
               int round, bool block_transpose) {
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
    auto *d0 = reference.part[0].get(), *d1 = reference.part[1].get();
    BatchedMap forward_map(forward_plan, h.rhs, even.sampled.ptr, odd.sampled.ptr);
    BatchedMap reverse_map(reverse_plan, h.rhs, d0->g.ptr, d1->g.ptr);
    BlockTranspose block_map(forward_map, d0->g.ptr, d1->g.ptr);
    const double one = 1., zero = 0.;
    auto dense_forward = [&] {
      for (DensePart *p : {d0, d1})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, h.rows, p->cols,
          &one, p->x.ptr, h.rhs, p->p.ptr, h.rows, &zero, p->out.ptr, h.rhs));
    };
    auto compressed = [&] {
      for (int parity = 0; parity < 2; ++parity) {
        Part *p = parity ? &odd : &even;
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, h.rhs, h.samples, p->cols,
          &one, reference.part[parity]->x.ptr, h.rhs, p->a.ptr, h.samples, &zero, p->sampled.ptr, h.rhs));
      }
    };
    auto interpolate_forward = [&] { forward_map.apply(blas); };
    auto candidate_forward = [&] { compressed(); interpolate_forward(); };
    auto dense_reverse = [&] {
      for (DensePart *p : {d0, d1})
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rhs, p->cols, h.rows,
          &one, p->g.ptr, h.rhs, p->p.ptr, h.rows, &zero, p->back.ptr, h.rhs));
    };
    auto gather = [&] {
      if (block_transpose) block_map.apply(blas, stream);
      else reverse_map.apply(blas);
    };
    auto reverse_gemm = [&] {
      for (int parity = 0; parity < 2; ++parity) {
        Part *p = parity ? &odd : &even;
        const double *g = block_transpose ? (parity ? block_map.odd.ptr : block_map.even.ptr) :
                                           (parity ? reverse_map.odd.ptr : reverse_map.even.ptr);
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
                          hemispheres(d0->out.host(), d1->out.host()), h.rhs);
    if (!block_transpose) { reverse_map.output(0); reverse_map.output(1); }
    auto actual = even.back.host(), odd_actual = odd.back.host();
    auto exact = d0->back.host(), odd_reference = d1->back.host();
    actual.insert(actual.end(), odd_actual.begin(), odd_actual.end());
    exact.insert(exact.end(), odd_reference.begin(), odd_reference.end());
    Error reverse = error(actual, exact, h.rhs);
    passed = valid(forward) && valid(reverse);
    const double dense_cost = 2. * h.rhs * h.rows * (even.cols + odd.cols);
    const double sample_cost = 2. * h.rhs * h.samples * (even.cols + odd.cols);
    auto cost = [&](const HostTiles &p) { return sample_cost + 4. * h.rhs * p.groups * p.tile * p.span; };
    const double reduction_adds = double(block_map.host.indices.size()) * h.rhs;
    const double adjoint_cost = block_transpose ? cost(forward_plan) + reduction_adds : cost(reverse_plan);
    size_t free, total;
    CHECK(cudaMemGetInfo(&free, &total));
    std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"m\":%d,\"tile\":%d,"
      "\"repeat\":%d,\"rows\":%d,\"samples\":%d,\"rhs\":%d,\"cols_even\":%d,\"cols_odd\":%d,"
      "\"adjoint_reduction_adds\":%.0f,"
      "\"forward_l2\":%.17g,\"forward_linf\":%.17g,\"adjoint_l2\":%.17g,"
      "\"adjoint_linf\":%.17g,\"forward_flops_ratio\":%.9g,\"adjoint_flops_ratio\":%.9g,"
      "\"forward_span\":%d,\"adjoint_span\":%d,\"pass\":%s,\"device_used_mib\":%.1f}\n",
      label, h.m, tile, round + 1, h.rows, h.samples, h.rhs, even.cols, odd.cols,
      block_transpose ? reduction_adds : 0.,
      forward.l2, forward.linf, reverse.l2, reverse.linf,
      cost(forward_plan) / dense_cost, adjoint_cost / dense_cost,
      forward_plan.span, block_transpose ? forward_plan.span : reverse_plan.span,
      passed ? "true" : "false", double(total - free) / 1048576.);
    if (passed) {
      Graph df(stream, dense_forward), cf(stream, candidate_forward);
      Graph dr(stream, dense_reverse), cr(stream, candidate_reverse);
      Graph cg(stream, compressed), ip(stream, interpolate_forward), ga(stream, gather), rg(stream, reverse_gemm);
      for (int repeat = round; repeat <= round; ++repeat) {
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
      std::printf("{\"kind\":\"stages\",\"case\":\"%s\",\"repeat\":%d,\"sample_gemm_ms\":%.9g,"
        "\"interpolation_ms\":%.9g,\"transpose_gather_ms\":%.9g,\"transpose_gemm_ms\":%.9g}\n",
        label, round + 1, timing(cg.exec, stream), timing(ip.exec, stream), timing(ga.exec, stream), timing(rg.exec, stream));
      if (block_transpose) {
        Graph pg(stream, [&] { block_map.multiply(blas); });
        Graph reduction(stream, [&] { block_map.reduce(stream); });
        std::printf("{\"kind\":\"overlap_stages\",\"case\":\"%s\",\"repeat\":%d,"
          "\"partial_gemm_ms\":%.9g,\"reduction_ms\":%.9g}\n",
          label, round + 1, timing(pg.exec, stream), timing(reduction.exec, stream));
      }
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
    if (argc - first != 12) throw std::runtime_error("three orders and four variants required");
    std::vector<Fixture> fixtures;
    std::vector<std::string> labels;
    for (int i = first; i < argc; ++i) {
      fixtures.emplace_back(argv[i]);
      std::string label(argv[i]);
      labels.push_back(label.substr(label.find_last_of('/') + 1) + "-tile32");
      const auto &f = fixtures.back();
      HostTiles forward(f, 32, false), reverse(f, 32, true);
      HostOverlap overlap(forward);
      std::printf("fixture=%s rows=%d samples=%d rhs=%d forward_span=%d reverse_span=%d packing=exact ok\n",
        argv[i], f.h.rows, f.h.samples, f.h.rhs, forward.span, reverse.span);
    }
    for (size_t i = 0; i < fixtures.size(); ++i) {
      const auto &base = fixtures[i / 4 * 4], &value = fixtures[i];
      if (base.h.m != value.h.m || base.h.rows != value.h.rows)
        throw std::runtime_error("comparison group shapes differ");
      for (int p = 0; p < 2; ++p)
        if (base.part[p].x != value.part[p].x || base.part[p].p != value.part[p].p ||
            base.part[p].g != value.part[p].g)
          throw std::runtime_error("comparison inputs or matrices differ");
      // Sample matrices differ because Jacobi and uniform grids have different nodes.
      // The reference matrix and all transform inputs above must remain identical.
    }
    if (inspect) return 0;
    // Keep dense controls and common input addresses fixed across variants and rounds.
    std::vector<std::unique_ptr<DenseReference>> references;
    for (size_t i = 0; i < fixtures.size(); i += 4)
      references.push_back(std::make_unique<DenseReference>(fixtures[i]));
    // Rotate orders and variant permutations; every variant gets a nearby dense control.
    const int permutations[8][4] = {{0,1,2,3},{1,2,3,0},{2,3,0,1},{3,0,1,2},
                                   {3,2,1,0},{0,3,2,1},{1,0,3,2},{2,1,0,3}};
    bool passed = true;
    for (int round = 0; round < 8; ++round) {
      for (int step = 0; step < 3; ++step) {
        int group = (step + round) % 3;
        for (int variant : permutations[round]) {
          int i = 4 * group + variant;
          std::string label = labels[i] + "-map";
          passed = run_tiled(fixtures[i], *references[group], 32, label.c_str(), round, false) && passed;
        }
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "tiled_probe: %s\n", e.what());
    return 2;
  }
}

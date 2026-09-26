// Split only exact zero padding; coefficients and nonoverlapping outputs stay fixed.
struct HostBand {
  int span;
  std::vector<int> batches, starts;
  std::vector<double> weights;
};

std::vector<HostBand> split_bands(const HostTiles &h, int cut) {
  if (cut < 1) throw std::runtime_error("invalid band cut");
  const int narrow = std::min(cut, h.span);
  std::vector<HostBand> bands{{narrow, {}, {}, {}}};
  if (narrow < h.span) bands.push_back({h.span, {}, {}, {}});
  std::vector<int> coverage(2 * h.groups, 0);
  for (int batch = 0; batch < 2 * h.groups; ++batch) {
    int lo = h.span, hi = -1;
    const double *old = h.weights.data() + size_t(batch) * h.span * h.tile;
    for (int j = 0; j < h.tile; ++j)
      for (int k = 0; k < h.span; ++k)
        if (old[j * h.span + k] != 0.) { lo = std::min(lo, k); hi = std::max(hi, k); }
    const int used = hi < lo ? 0 : hi - lo + 1;
    auto &band = bands[used <= narrow ? 0 : 1];
    const int start = hi < lo ? 0 : std::min(h.starts[batch] + lo, h.input_rows - band.span);
    const size_t offset = band.weights.size();
    band.weights.resize(offset + size_t(band.span) * h.tile, 0.);
    for (int j = 0; j < h.tile; ++j) {
      for (int k = 0; k < h.span; ++k) {
        const int col = h.starts[batch] + k - start;
        const double value = old[j * h.span + k];
        if (col < 0 || col >= band.span) {
          if (value != 0.) throw std::runtime_error("band dropped a coefficient");
        } else {
          band.weights[offset + size_t(j) * band.span + col] = value;
        }
      }
      for (int col = 0; col < h.input_rows; ++col) {
        const int a = col - h.starts[batch], b = col - start;
        const double before = a >= 0 && a < h.span ? old[j * h.span + a] : 0.;
        const double after = b >= 0 && b < band.span ? band.weights[offset + j * band.span + b] : 0.;
        if (before != after) throw std::runtime_error("band map differs from original");
      }
    }
    band.batches.push_back(batch);
    band.starts.push_back(start);
    ++coverage[batch];
  }
  if (std::any_of(coverage.begin(), coverage.end(), [](int n) { return n != 1; }))
    throw std::runtime_error("band outputs are missing or overlap");
  bands.erase(std::remove_if(bands.begin(), bands.end(), [](const HostBand &b) {
    return b.batches.empty();
  }), bands.end());
  return bands;
}

void test_band_edges(const HostTiles &example) {
  int count = 0;
  for (int input : {1, 7, 159, 160, 161, 257}) {
    for (int output : {1, 31, 32, 33, 65}) {
      for (int pattern = 0; pattern < 3; ++pattern) {
        HostTiles h = example;
        h.input_rows = input; h.output_rows = output; h.tile = 32;
        h.groups = (output + 31) / 32; h.padded_rows = h.groups * 32;
        h.span = std::min(input, 256);
        h.starts.resize(2 * h.groups);
        h.weights.assign(size_t(2) * h.groups * h.span * h.tile, 0.);
        for (int batch = 0; batch < 2 * h.groups; ++batch) {
          h.starts[batch] = batch % (input - h.span + 1);
          for (int j = 0; j < h.tile && (batch % h.groups) * 32 + j < output; ++j) {
            if (pattern == 0 || (pattern == 1 && batch % 2 == 0)) continue;
            for (int k = 0; k < h.span; ++k)
              if (pattern == 2 || k == h.span - 1)
                h.weights[size_t(batch) * h.span * h.tile + j * h.span + k] = (k % 7 - 3) * .125;
          }
        }
        split_bands(h, 160);
        ++count;
      }
    }
  }
  std::printf("band_edge_cases=%d packing=exact coverage=once\n", count);
}

struct DeviceBand {
  int span, count;
  Buffer<double> weights;
  Buffer<const double *> input_ptrs, weight_ptrs;
  Buffer<double *> output_ptrs;
  DeviceBand(const HostBand &h, const HostTiles &plan, int rhs,
             const double *a, const double *b, double *even, double *odd)
      : span(h.span), count(int(h.batches.size())), weights(h.weights),
        input_ptrs(h.batches.size()), weight_ptrs(h.batches.size()), output_ptrs(h.batches.size()) {
    std::vector<const double *> inputs, matrices;
    std::vector<double *> outputs;
    for (int i = 0; i < count; ++i) {
      const int batch = h.batches[i], parity = batch / plan.groups, group = batch % plan.groups;
      inputs.push_back((parity ? b : a) + size_t(rhs) * h.starts[i]);
      matrices.push_back(weights.ptr + size_t(i) * span * plan.tile);
      outputs.push_back((parity ? odd : even) + size_t(group) * plan.tile * rhs);
    }
    CHECK(cudaMemcpy(input_ptrs.ptr, inputs.data(), inputs.size() * sizeof(inputs[0]), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(weight_ptrs.ptr, matrices.data(), matrices.size() * sizeof(matrices[0]), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(output_ptrs.ptr, outputs.data(), outputs.size() * sizeof(outputs[0]), cudaMemcpyHostToDevice));
  }
};

struct BandedMap {
  const HostTiles &h;
  int rhs;
  Buffer<double> even, odd;
  std::vector<std::unique_ptr<DeviceBand>> bands;
  double flops = 0;
  BandedMap(const HostTiles &host, int fields, const double *a, const double *b)
      : h(host), rhs(fields), even(size_t(rhs) * h.padded_rows), odd(size_t(rhs) * h.padded_rows) {
    for (const auto &band : split_bands(h, 160)) {
      bands.push_back(std::make_unique<DeviceBand>(band, h, rhs, a, b, even.ptr, odd.ptr));
      flops += 2. * rhs * h.tile * band.span * band.batches.size();
    }
  }
  void apply(cublasHandle_t blas) {
    const double one = 1., zero = 0.;
    for (auto &band : bands)
      CHECK(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, rhs, h.tile, band->span,
        &one, band->input_ptrs.ptr, rhs, band->weight_ptrs.ptr, band->span, &zero,
        band->output_ptrs.ptr, rhs, band->count));
  }
  void check_padding() const {
    for (const auto *part : {&even, &odd}) {
      const auto values = part->host();
      for (size_t i = size_t(rhs) * h.output_rows; i < values.size(); ++i)
        if (values[i] != 0.) throw std::runtime_error("nonzero banded padding");
    }
  }
};

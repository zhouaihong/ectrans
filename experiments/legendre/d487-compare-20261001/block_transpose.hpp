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


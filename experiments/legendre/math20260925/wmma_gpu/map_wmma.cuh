#include <mma.h>

// Same packed matrix as cuBLAS; shared memory handles the unaligned rhs=242 stride.
template<int Fields>
__global__ void map_fp64(const double *a0, const double *a1, const double *weights,
                         const int *starts, double *out0, double *out1,
                         int rhs, int span, int groups) {
  namespace w = nvcuda::wmma;
  constexpr int Chunk = 32, Tile = 32;
  __shared__ __align__(32) double a[Fields * Chunk];
  __shared__ __align__(32) double b[Chunk * Tile];
  __shared__ __align__(32) double c[Fields * Tile];
  int parity = blockIdx.z, group = blockIdx.y, first = blockIdx.x * Fields;
  int batch = parity * groups + group, start = starts[batch];
  const double *input = parity ? a1 : a0;
  const double *matrix = weights + size_t(batch) * span * Tile;
  double *output = parity ? out1 : out0;
  int warp = threadIdx.x / 32, field = (warp / 4) * 8, column = (warp % 4) * 8;
  w::fragment<w::matrix_a, 8, 8, 4, double, w::col_major> af;
  w::fragment<w::matrix_b, 8, 8, 4, double, w::col_major> bf;
  w::fragment<w::accumulator, 8, 8, 4, double> acc;
  w::fill_fragment(acc, 0.);
  for (int base = 0; base < span; base += Chunk) {
    for (int i = threadIdx.x; i < Fields * Chunk; i += blockDim.x) {
      int r = i % Fields, k = base + i / Fields;
      a[i] = first + r < rhs && k < span ? input[size_t(start + k) * rhs + first + r] : 0.;
    }
    for (int i = threadIdx.x; i < Chunk * Tile; i += blockDim.x) {
      int k = base + i % Chunk, col = i / Chunk;
      b[i] = k < span ? matrix[size_t(col) * span + k] : 0.;
    }
    __syncthreads();
    for (int k = 0; k < Chunk && base + k < span; k += 4) {
      w::load_matrix_sync(af, a + field + k * Fields, Fields);
      w::load_matrix_sync(bf, b + k + column * Chunk, Chunk);
      w::mma_sync(acc, af, bf, acc);
    }
    __syncthreads();
  }
  w::store_matrix_sync(c + field + column * Fields, acc, Fields, w::mem_col_major);
  __syncthreads();
  for (int i = threadIdx.x; i < Fields * Tile; i += blockDim.x) {
    int r = i % Fields, col = i / Fields;
    if (first + r < rhs) output[size_t(group * Tile + col) * rhs + first + r] = c[i];
  }
}

void launch_map(int fields, cudaStream_t stream, const double *a0, const double *a1,
                const double *weights, const int *starts, double *out0, double *out1,
                int rhs, int span, int groups) {
  dim3 grid((rhs + fields - 1) / fields, groups, 2);
  switch (fields) {
    case 8: map_fp64<8><<<grid, 128, 0, stream>>>(a0,a1,weights,starts,out0,out1,rhs,span,groups); break;
    case 16: map_fp64<16><<<grid, 256, 0, stream>>>(a0,a1,weights,starts,out0,out1,rhs,span,groups); break;
    case 32: map_fp64<32><<<grid, 512, 0, stream>>>(a0,a1,weights,starts,out0,out1,rhs,span,groups); break;
    default: throw std::runtime_error("invalid fp64 field tile");
  }
  CHECK(cudaGetLastError());
}

void test_map_fp64() {
  int cases = 0;
  for (int rhs : {1,7,8,9,15,16,17,242}) {
    for (int span : {1,3,4,7,8,31,32,33,80,192}) {
      constexpr int groups = 2, tile = 32;
      int input_rows = span + 3;
      std::vector<int> starts{0, 3, 1, 2};
      std::vector<double> x[2], reference[2], weights(size_t(2) * groups * span * tile);
      for (int p = 0; p < 2; ++p) {
        x[p].resize(size_t(input_rows) * rhs);
        reference[p].resize(size_t(groups * tile) * rhs, 0.);
        for (size_t i = 0; i < x[p].size(); ++i) x[p][i] = std::sin(.17 * i + p);
        for (int g = 0; g < groups; ++g) {
          int batch = p * groups + g;
          for (int col = 0; col < tile; ++col) {
            for (int k = 0; k < span; ++k) {
              double value = g * tile + col < 37 ? std::cos(.13 * (k + 3*col + 5*p)) / span : 0.;
              weights[size_t(batch) * span * tile + col * span + k] = value;
              for (int f = 0; f < rhs; ++f)
                reference[p][size_t(g*tile+col)*rhs+f] += value * x[p][size_t(starts[batch]+k)*rhs+f];
            }
          }
        }
      }
      Buffer<double> a0(x[0]), a1(x[1]), matrix(weights);
      Buffer<int> offsets(starts);
      Buffer<double> b0(reference[0].size()), b1(reference[1].size());
      for (int fields : {8,16,32}) {
        CHECK(cudaMemset(b0.ptr, 0xff, b0.size * sizeof(double)));
        CHECK(cudaMemset(b1.ptr, 0xff, b1.size * sizeof(double)));
        launch_map(fields, nullptr, a0.ptr, a1.ptr, matrix.ptr, offsets.ptr,
                   b0.ptr, b1.ptr, rhs, span, groups);
        CHECK(cudaDeviceSynchronize());
        auto actual0 = b0.host(), actual1 = b1.host();
        if (!valid(error(actual0, reference[0], rhs)) || !valid(error(actual1, reference[1], rhs)))
          throw std::runtime_error("fp64 map selftest mismatch");
        for (size_t i = size_t(37)*rhs; i < actual0.size(); ++i)
          if (actual0[i] != 0. || actual1[i] != 0.) throw std::runtime_error("padding selftest");
        ++cases;
      }
    }
  }
  std::printf("fp64_map_selftest cases=%d pass=true\n", cases);
}

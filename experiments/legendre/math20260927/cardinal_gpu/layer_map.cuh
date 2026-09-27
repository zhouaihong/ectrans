#include <mma.h>

struct LayerTask { int out, rows, k, indices, weights; };
struct LayerTile { int task, row; };

__device__ double layer_input(int row, int field, const double *work,
                              const double *g0, const double *g1, int rhs, int rows) {
  if (row >= 0) return work[size_t(row) * rhs + field];
  row = -1 - row;
  return (row < rows ? g0 : g1)[size_t(row % rows) * rhs + field];
}

__global__ void layer_simt(const LayerTask *tasks, const LayerTile *tiles,
                            const int *indices, const double *weights, double *work,
                            const double *g0, const double *g1, int rhs, int rows) {
  const auto tile = tiles[blockIdx.y];
  const auto task = tasks[tile.task];
  const int field = blockIdx.x * 32 + threadIdx.x;
  if (field >= rhs) return;
  for (int j = tile.row; j < min(tile.row + 8, task.rows); ++j) {
    double sum = 0.;
    for (int k = 0; k < task.k; ++k)
      sum = fma(weights[task.weights + j * task.k + k],
        layer_input(indices[task.indices + k], field, work, g0, g1, rhs, rows), sum);
    work[size_t(task.out + j) * rhs + field] = sum;
  }
}

// Each output interval has a single writer. Dependencies cross kernel boundaries.
template<int Fields, int Tile>
__global__ void layer_wmma(const LayerTask *tasks, const LayerTile *tiles,
                            const int *indices, const double *weights, double *work,
                            const double *g0, const double *g1, int rhs, int rows) {
  using namespace nvcuda;
  __shared__ __align__(32) double a[Fields * 32];
  __shared__ __align__(32) double b[32 * Tile];
  __shared__ __align__(32) double c[Fields * Tile];
  const auto tile = tiles[blockIdx.y];
  const auto task = tasks[tile.task];
  const int warp_field = (threadIdx.x / 32 % (Fields / 8)) * 8;
  const int warp_column = (threadIdx.x / 32 / (Fields / 8)) * 8;
  const int first_field = blockIdx.x * Fields;
  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::col_major> af;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> cf;
  wmma::fill_fragment(cf, 0.);
  for (int start = 0; start < task.k; start += 32) {
    for (int q = threadIdx.x; q < Fields * 32; q += blockDim.x) {
      const int field = first_field + q % Fields, k = start + q / Fields;
      a[q] = field < rhs && k < task.k ?
        layer_input(indices[task.indices + k], field, work, g0, g1, rhs, rows) : 0.;
    }
    for (int q = threadIdx.x; q < 32 * Tile; q += blockDim.x) {
      const int k = start + q % 32, j = tile.row + q / 32;
      b[q] = k < task.k && j < task.rows ? weights[task.weights + j * task.k + k] : 0.;
    }
    __syncthreads();
    for (int k = 0; k < 32 && start + k < task.k; k += 4) {
      wmma::load_matrix_sync(af, a + warp_field + k * Fields, Fields);
      wmma::load_matrix_sync(bf, b + warp_column * 32 + k, 32);
      wmma::mma_sync(cf, af, bf, cf);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(c + warp_field + warp_column * Fields, cf, Fields, wmma::mem_col_major);
  __syncthreads();
  for (int q = threadIdx.x; q < Fields * Tile; q += blockDim.x) {
    const int field = first_field + q % Fields, row = tile.row + q / Fields;
    if (field < rhs && row < task.rows) work[size_t(task.out + row) * rhs + field] = c[q];
  }
}

struct HostLayer {
  std::vector<LayerTask> tasks;
  std::vector<LayerTile> tiles;
  std::vector<int> indices;
  std::vector<double> weights;
};

std::vector<LayerTile> output_tiles(const HostLayer &h, int width) {
  std::vector<LayerTile> result;
  for (size_t t = 0; t < h.tasks.size(); ++t)
    for (int j = 0; j < h.tasks[t].rows; j += width) result.push_back({int(t),j});
  return result;
}

struct DeviceLayer {
  Buffer<LayerTask> tasks;
  Buffer<LayerTile> tiles;
  Buffer<LayerTile> tiles16, tiles32;
  Buffer<int> indices;
  Buffer<double> weights;
  explicit DeviceLayer(const HostLayer &h)
      : tasks(h.tasks), tiles(h.tiles), tiles16(output_tiles(h,16)), tiles32(output_tiles(h,32)),
        indices(h.indices), weights(h.weights) {}
  void apply(int mode, cudaStream_t stream, double *work,
             const double *g0, const double *g1, int rhs, int rows) {
    auto &selected = mode == 3 ? tiles16 : mode == 4 ? tiles32 : tiles;
    dim3 grid((rhs + (mode == 1 ? 7 : 31)) / (mode == 1 ? 8 : 32), selected.size);
    if (mode == 0)
      layer_simt<<<grid, 32, 0, stream>>>(tasks.ptr, tiles.ptr, indices.ptr, weights.ptr, work, g0, g1, rhs, rows);
    else if (mode == 1)
      layer_wmma<8,8><<<grid, 32, 0, stream>>>(tasks.ptr, tiles.ptr, indices.ptr, weights.ptr, work, g0, g1, rhs, rows);
    else if (mode == 2)
      layer_wmma<32,8><<<grid, 128, 0, stream>>>(tasks.ptr, tiles.ptr, indices.ptr, weights.ptr, work, g0, g1, rhs, rows);
    else if (mode == 3)
      layer_wmma<32,16><<<grid, 256, 0, stream>>>(tasks.ptr, selected.ptr, indices.ptr, weights.ptr, work, g0, g1, rhs, rows);
    else
      layer_wmma<32,32><<<grid, 512, 0, stream>>>(tasks.ptr, selected.ptr, indices.ptr, weights.ptr, work, g0, g1, rhs, rows);
    CHECK(cudaGetLastError());
  }
};

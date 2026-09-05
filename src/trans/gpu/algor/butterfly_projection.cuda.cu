// (C) Copyright 2026- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0.

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int fields_per_block = 16;
constexpr int outputs_per_block = 8;
constexpr int gemm_tile = 16;
constexpr int warp_threads = 32;
constexpr int wmma_field_tile = 64;
constexpr int wmma_warps_per_block = 16;

__global__ void butterfly_grouped_gemm_kernel(
    const double *__restrict__ input, const double *__restrict__ factors,
    double *__restrict__ output, int leading_dimension,
    const int *__restrict__ outputs, const int *__restrict__ inner,
    const int *__restrict__ factor_ld, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ factor_offsets, std::int64_t add_src,
    std::int64_t add_dst, int group_base, int factor_transposed,
    int atomic_output, int compact_projection, double beta) {
  __shared__ __align__(32) double tile_a[wmma_field_tile * 4];
  __shared__ __align__(32) double tile_b[4 * gemm_tile];
  __shared__ __align__(32) double tile_c[wmma_field_tile * gemm_tile];

  const int group = group_base + static_cast<int>(blockIdx.z);
  const int thread = threadIdx.x;
  const int n = outputs[group];
  const int k = inner[group] - (compact_projection ? n : 0);
  const int ldb = compact_projection ? n : factor_ld[group];
  const std::int64_t factor_offset = factor_offsets[group];

  const int warp = thread / warp_threads;
  const int warp_row = (warp & 7) * 8;
  const int warp_column = (warp >> 3) * 8;
  for (int field_base = 0; field_base < leading_dimension;
       field_base += wmma_field_tile) {
    for (int column_base = 0; column_base < n; column_base += gemm_tile) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double,
                             nvcuda::wmma::col_major>
          a_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double,
                             nvcuda::wmma::col_major>
          b_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>
          accumulator;
      nvcuda::wmma::fill_fragment(accumulator, 0.0);

      for (int k_base = 0; k_base < k; k_base += 4) {
        for (int index = thread; index < wmma_field_tile * 4;
             index += blockDim.x) {
          const int row = index % wmma_field_tile;
          const int local_k = index / wmma_field_tile;
          const int field = field_base + row;
          const int global_k = k_base + local_k;
          tile_a[index] =
              field < leading_dimension && global_k < k
                  ? input[(src[group] + add_src + global_k) *
                              leading_dimension +
                          field]
                  : 0.0;
        }
        for (int index = thread; index < 4 * gemm_tile;
             index += blockDim.x) {
          const int b_k = factor_transposed ? index / gemm_tile : index % 4;
          const int local_column =
              factor_transposed ? index % gemm_tile : index / 4;
          const int column = column_base + local_column;
          tile_b[b_k + local_column * 4] =
              k_base + b_k < k && column < n
                  ? (factor_transposed
                         ? factors[factor_offset + column +
                                   static_cast<std::int64_t>(k_base + b_k) *
                                       ldb]
                         : factors[factor_offset + k_base + b_k +
                                   static_cast<std::int64_t>(column) * ldb])
                  : 0.0;
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_fragment, tile_a + warp_row,
                                       wmma_field_tile);
        nvcuda::wmma::load_matrix_sync(b_fragment,
                                       tile_b + warp_column * 4, 4);
        nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment,
                               accumulator);
        __syncthreads();
      }

      nvcuda::wmma::store_matrix_sync(
          tile_c + warp_row + warp_column * wmma_field_tile, accumulator,
          wmma_field_tile, nvcuda::wmma::mem_col_major);
      __syncthreads();

      for (int index = thread; index < wmma_field_tile * gemm_tile;
           index += blockDim.x) {
        const int row = index % wmma_field_tile;
        const int local_column = index / wmma_field_tile;
        const int field = field_base + row;
        const int column = column_base + local_column;
        if (field < leading_dimension && column < n) {
          double *target = output +
                           (dst[group] + add_dst + column) *
                               leading_dimension +
                           field;
          const double value = tile_c[index];
          if (atomic_output) {
            atomicAdd(target, value);
          } else {
            *target = beta == 0.0 ? value : value + beta * *target;
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void butterfly_compact_projection_kernel(
    const double *__restrict__ input, const double *__restrict__ factors,
    double *__restrict__ output, int leading_dimension,
    const int *__restrict__ ranks, const int *__restrict__ columns,
    const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ pivot_offsets,
    const std::int64_t *__restrict__ factor_offsets,
    const int *__restrict__ pivots, std::int64_t add_src,
    std::int64_t add_dst, int group_base, double beta) {
  __shared__ __align__(32) double tile_a[wmma_field_tile * 4];
  __shared__ __align__(32) double tile_b[4 * gemm_tile];
  __shared__ __align__(32) double tile_c[wmma_field_tile * gemm_tile];

  const int group = group_base + static_cast<int>(blockIdx.z);
  const int thread = threadIdx.x;
  const int n = ranks[group];
  const int k = columns[group] - n;

  const int warp = thread / warp_threads;
  const int warp_row = (warp & 7) * 8;
  const int warp_column = (warp >> 3) * 8;
  const std::int64_t pivot_offset = pivot_offsets[group];
  const std::int64_t factor_offset = factor_offsets[group];
  for (int field_base = 0; field_base < leading_dimension;
       field_base += wmma_field_tile) {
    for (int column_base = 0; column_base < n; column_base += gemm_tile) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double,
                             nvcuda::wmma::col_major>
          a_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double,
                             nvcuda::wmma::col_major>
          b_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>
          accumulator;
      nvcuda::wmma::fill_fragment(accumulator, 0.0);

      for (int k_base = 0; k_base < k; k_base += 4) {
        for (int index = thread; index < wmma_field_tile * 4;
             index += blockDim.x) {
          const int row = index % wmma_field_tile;
          const int local_k = index / wmma_field_tile;
          const int field = field_base + row;
          const int global_k = k_base + local_k;
          tile_a[index] =
              field < leading_dimension && global_k < k
                  ? input[(src[group] + add_src +
                           pivots[pivot_offset + n + global_k]) *
                              leading_dimension +
                          field]
                  : 0.0;
        }
        for (int index = thread; index < 4 * gemm_tile;
             index += blockDim.x) {
          const int b_k = index / gemm_tile;
          const int local_column = index % gemm_tile;
          const int column = column_base + local_column;
          tile_b[b_k + local_column * 4] =
              k_base + b_k < k && column < n
                  ? factors[factor_offset + column +
                            static_cast<std::int64_t>(k_base + b_k) * n]
                  : 0.0;
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_fragment, tile_a + warp_row,
                                       wmma_field_tile);
        nvcuda::wmma::load_matrix_sync(b_fragment,
                                       tile_b + warp_column * 4, 4);
        nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment,
                               accumulator);
        __syncthreads();
      }

      nvcuda::wmma::store_matrix_sync(
          tile_c + warp_row + warp_column * wmma_field_tile, accumulator,
          wmma_field_tile, nvcuda::wmma::mem_col_major);
      __syncthreads();

      for (int index = thread; index < wmma_field_tile * gemm_tile;
           index += blockDim.x) {
        const int row = index % wmma_field_tile;
        const int local_column = index / wmma_field_tile;
        const int field = field_base + row;
        const int column = column_base + local_column;
        if (field < leading_dimension && column < n) {
          const double pivot =
              input[(src[group] + add_src +
                     pivots[pivot_offset + column]) *
                        leading_dimension +
                    field];
          double *target = output +
                           (dst[group] + add_dst + column) *
                               leading_dimension +
                           field;
          const double value = pivot + tile_c[index];
          *target = beta == 0.0 ? value : value + beta * *target;
        }
      }
      __syncthreads();
    }
  }
}

void launch_grouped_gemm(
    const double *input, const double *factors, double *output,
    int leading_dimension, const int *outputs, const int *inner,
    const int *factor_ld, const std::int64_t *src, const std::int64_t *dst,
    const std::int64_t *factor_offsets, std::int64_t add_src,
    std::int64_t add_dst, int group_count, int max_outputs,
    int factor_transposed, int atomic_output, int compact_projection,
    double beta, std::intptr_t stream_value) {
  if (leading_dimension <= 0 || group_count <= 0 || max_outputs <= 0)
    return;

  const dim3 block(wmma_warps_per_block * warp_threads);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  constexpr int max_grid_z = 65535;
  for (int group_base = 0; group_base < group_count;
       group_base += max_grid_z) {
    const int groups =
        group_count - group_base < max_grid_z ? group_count - group_base
                                               : max_grid_z;
    const dim3 grid(1, 1, static_cast<unsigned int>(groups));
    butterfly_grouped_gemm_kernel<<<grid, block, 0, stream>>>(
        input, factors, output, leading_dimension, outputs, inner, factor_ld,
        src, dst, factor_offsets, add_src, add_dst, group_base,
        factor_transposed, atomic_output, compact_projection, beta);
  }
}

void launch_compact_projection(
    const double *input, const double *factors, double *output,
    int leading_dimension, const int *ranks, const int *columns,
    const std::int64_t *src, const std::int64_t *dst,
    const std::int64_t *pivot_offsets,
    const std::int64_t *factor_offsets, const int *pivots,
    std::int64_t add_src, std::int64_t add_dst, int group_count,
    int max_rank, double beta, std::intptr_t stream_value) {
  if (leading_dimension <= 0 || group_count <= 0 || max_rank <= 0)
    return;

  const dim3 block(wmma_warps_per_block * warp_threads);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  constexpr int max_grid_z = 65535;
  for (int group_base = 0; group_base < group_count;
       group_base += max_grid_z) {
    const int groups =
        group_count - group_base < max_grid_z ? group_count - group_base
                                               : max_grid_z;
    const dim3 grid(1, 1, static_cast<unsigned int>(groups));
    butterfly_compact_projection_kernel<<<grid, block, 0, stream>>>(
        input, factors, output, leading_dimension, ranks, columns, src, dst,
        pivot_offsets, factor_offsets, pivots, add_src, add_dst, group_base,
        beta);
  }
}

__global__ void butterfly_projection_forward_kernel(
    const double *__restrict__ input, double *__restrict__ output,
    int leading_dimension, const int *__restrict__ ranks,
    const int *__restrict__ columns, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ pivot_offsets,
    const std::int64_t *__restrict__ scratch_offsets,
    const int *__restrict__ tile_groups,
    const int *__restrict__ tile_starts,
    const int *__restrict__ pivots, std::int64_t add_src,
    std::int64_t add_dst, std::int64_t add_scratch, double beta) {
  const int field_lane = threadIdx.x;
  const int column_lane = threadIdx.y;
  const int group = tile_groups[blockIdx.x];
  const int column = tile_starts[blockIdx.x] + column_lane;
  const int field = static_cast<int>(blockIdx.y) * fields_per_block + field_lane;
  const int rank = ranks[group];
  const int column_count = columns[group];
  if (column >= column_count || field >= leading_dimension)
    return;

  const std::int64_t pivot_base = pivot_offsets[group];
  const double value =
      input[(src[group] + add_src + pivots[pivot_base + column]) *
                leading_dimension +
            field];
  if (column < rank) {
    double *target =
        output + (dst[group] + add_dst + column) * leading_dimension + field;
    *target = beta == 0.0 ? value : value + beta * *target;
  } else {
    output[(add_scratch + scratch_offsets[group] + column - rank) *
               leading_dimension +
           field] = value;
  }
}

__global__ void butterfly_projection_reverse_kernel(
    const double *__restrict__ input, double *__restrict__ output,
    int leading_dimension, const int *__restrict__ ranks,
    const int *__restrict__ columns, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ pivot_offsets,
    const std::int64_t *__restrict__ scratch_offsets,
    const int *__restrict__ tile_groups,
    const int *__restrict__ tile_starts,
    const int *__restrict__ pivots, std::int64_t add_src,
    std::int64_t add_dst, std::int64_t add_scratch, double beta) {
  const int field_lane = threadIdx.x;
  const int column_lane = threadIdx.y;
  const int group = tile_groups[blockIdx.x];
  const int column = tile_starts[blockIdx.x] + column_lane;
  const int field = static_cast<int>(blockIdx.y) * fields_per_block + field_lane;
  const int column_count = columns[group];
  const int rank = ranks[group];
  if (column >= column_count || field >= leading_dimension)
    return;

  const double value =
      column < rank
          ? input[(dst[group] + add_src + column) * leading_dimension + field]
          : output[(add_scratch + scratch_offsets[group] + column - rank) *
                       leading_dimension +
                   field];
  const std::int64_t pivot_base = pivot_offsets[group];
  double *target =
      output + (src[group] + add_dst + pivots[pivot_base + column]) *
                   leading_dimension +
      field;
  if (beta == 0.0) {
    *target = value;
  } else if (beta == 1.0) {
    atomicAdd(target, value);
  } else {
    *target = value + beta * *target;
  }
}

} // namespace

extern "C" void ectrans_butterfly_projection_cuda(
    const double *input, double *output, int leading_dimension,
    const int *ranks, const int *columns, const std::int64_t *src,
    const std::int64_t *dst, const std::int64_t *pivot_offsets,
    const std::int64_t *scratch_offsets, const int *tile_groups,
    const int *tile_starts, const int *pivots, std::int64_t add_src,
    std::int64_t add_dst, std::int64_t add_scratch, int tile_count,
    int reverse, double beta, std::intptr_t stream_value) {
  if (leading_dimension <= 0 || tile_count <= 0)
    return;

  const dim3 block(fields_per_block, outputs_per_block);
  const dim3 grid(static_cast<unsigned int>(tile_count),
                  static_cast<unsigned int>((leading_dimension +
                                             fields_per_block - 1) /
                                            fields_per_block));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  if (reverse) {
    butterfly_projection_reverse_kernel<<<grid, block, 0, stream>>>(
        input, output, leading_dimension, ranks, columns, src, dst,
        pivot_offsets, scratch_offsets, tile_groups, tile_starts, pivots,
        add_src, add_dst, add_scratch, beta);
  } else {
    butterfly_projection_forward_kernel<<<grid, block, 0, stream>>>(
        input, output, leading_dimension, ranks, columns, src, dst,
        pivot_offsets, scratch_offsets, tile_groups, tile_starts, pivots,
        add_src, add_dst, add_scratch, beta);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "butterfly projection CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

extern "C" void ectrans_butterfly_grouped_gemm_cuda(
    const double *input, const double *factors, double *output,
    int leading_dimension, const int *outputs, const int *inner,
    const int *factor_ld, const std::int64_t *src, const std::int64_t *dst,
    const std::int64_t *factor_offsets, std::int64_t add_src,
    std::int64_t add_dst, int group_count, int max_outputs,
    int factor_transposed, int atomic_output, double beta,
    std::intptr_t stream_value) {
  launch_grouped_gemm(input, factors, output, leading_dimension, outputs,
                      inner, factor_ld, src, dst, factor_offsets, add_src,
                      add_dst, group_count, max_outputs, factor_transposed,
                      atomic_output, 0, beta, stream_value);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "butterfly grouped GEMM CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

extern "C" void ectrans_butterfly_compact_projection_cuda(
    const double *input, const double *factors, double *output,
    int leading_dimension, const int *ranks, const int *columns,
    const std::int64_t *src, const std::int64_t *dst,
    const std::int64_t *pivot_offsets,
    const std::int64_t *factor_offsets, const int *pivots,
    std::int64_t add_src, std::int64_t add_dst, int group_count, int max_rank,
    double beta, std::intptr_t stream_value) {
  launch_compact_projection(input, factors, output, leading_dimension, ranks,
                            columns, src, dst, pivot_offsets, factor_offsets,
                            pivots, add_src, add_dst, group_count, max_rank, beta,
                            stream_value);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr,
                 "butterfly compact projection CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

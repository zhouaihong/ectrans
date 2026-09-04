// (C) Copyright 2026- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int fields_per_block = 16;
constexpr int outputs_per_block = 8;
constexpr int gemm_tile = 16;

__global__ void butterfly_grouped_gemm_kernel(
    const double *__restrict__ input, const double *__restrict__ factors,
    double *__restrict__ output, int leading_dimension,
    const int *__restrict__ outputs, const int *__restrict__ inner,
    const int *__restrict__ factor_ld, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ factor_offsets, std::int64_t add_src,
    std::int64_t add_dst, int group_base, int factor_transposed,
    double beta) {
  __shared__ double tile_a[gemm_tile][gemm_tile + 1];
  __shared__ double tile_b[gemm_tile][gemm_tile + 1];

  const int group = group_base + static_cast<int>(blockIdx.z);
  const int field = static_cast<int>(blockIdx.x) * gemm_tile + threadIdx.x;
  const int column = static_cast<int>(blockIdx.y) * gemm_tile + threadIdx.y;
  const int n = outputs[group];
  const int k = inner[group];
  if (static_cast<int>(blockIdx.y) * gemm_tile >= n)
    return;

  double value = 0.0;
  for (int k_base = 0; k_base < k; k_base += gemm_tile) {
    const int a_k = k_base + threadIdx.y;
    tile_a[threadIdx.x][threadIdx.y] =
        field < leading_dimension && a_k < k
            ? input[(src[group] + add_src + a_k) * leading_dimension + field]
            : 0.0;

    const int b_k = k_base + threadIdx.x;
    double factor = 0.0;
    if (b_k < k && column < n) {
      const std::int64_t offset = factor_offsets[group];
      factor = factor_transposed
                   ? factors[offset + column +
                             static_cast<std::int64_t>(b_k) * factor_ld[group]]
                   : factors[offset + b_k +
                             static_cast<std::int64_t>(column) * factor_ld[group]];
    }
    tile_b[threadIdx.x][threadIdx.y] = factor;
    __syncthreads();

#pragma unroll
    for (int q = 0; q < gemm_tile; ++q)
      value += tile_a[threadIdx.x][q] * tile_b[q][threadIdx.y];
    __syncthreads();
  }

  if (field < leading_dimension && column < n) {
    double *target =
        output + (dst[group] + add_dst + column) * leading_dimension + field;
    *target = beta == 0.0 ? value : value + beta * *target;
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
    int factor_transposed, double beta, std::intptr_t stream_value) {
  if (leading_dimension <= 0 || group_count <= 0 || max_outputs <= 0)
    return;

  const dim3 block(gemm_tile, gemm_tile);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  constexpr int max_grid_z = 65535;
  for (int group_base = 0; group_base < group_count;
       group_base += max_grid_z) {
    const int groups =
        group_count - group_base < max_grid_z ? group_count - group_base
                                               : max_grid_z;
    const dim3 grid(
        static_cast<unsigned int>((leading_dimension + gemm_tile - 1) /
                                  gemm_tile),
        static_cast<unsigned int>((max_outputs + gemm_tile - 1) / gemm_tile),
        static_cast<unsigned int>(groups));
    butterfly_grouped_gemm_kernel<<<grid, block, 0, stream>>>(
        input, factors, output, leading_dimension, outputs, inner, factor_ld,
        src, dst, factor_offsets, add_src, add_dst, group_base,
        factor_transposed, beta);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "butterfly grouped GEMM CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

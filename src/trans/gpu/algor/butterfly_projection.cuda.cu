// (C) Copyright 2026- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int warp_size = 32;
constexpr int warps_per_block = 4;

__global__ void butterfly_projection_forward_kernel(
    const double *__restrict__ input, double *__restrict__ output,
    int leading_dimension, const int *__restrict__ ranks,
    const int *__restrict__ columns, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ factor_offsets,
    const std::int64_t *__restrict__ pivot_offsets,
    const int *__restrict__ tile_groups,
    const int *__restrict__ tile_starts,
    const int *__restrict__ pivots, const double *__restrict__ factors,
    std::int64_t add_src, std::int64_t add_dst, double beta) {
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const int group = tile_groups[blockIdx.x];
  const int output_column = tile_starts[blockIdx.x] + warp;
  const int field = static_cast<int>(blockIdx.y) * warp_size + lane;
  const int rank = ranks[group];
  if (output_column >= rank || field >= leading_dimension)
    return;

  const int column_count = columns[group];
  const std::int64_t pivot_base = pivot_offsets[group];
  const std::int64_t factor_base = factor_offsets[group];
  double value = input[(src[group] + add_src +
                        pivots[pivot_base + output_column]) *
                           leading_dimension +
                       field];
  for (int column = rank; column < column_count; ++column) {
    const double factor =
        factors[factor_base + static_cast<std::int64_t>(column - rank) * rank +
                output_column];
    value = fma(input[(src[group] + add_src + pivots[pivot_base + column]) *
                          leading_dimension +
                      field],
                factor, value);
  }

  double *target =
      output + (dst[group] + add_dst + output_column) * leading_dimension +
      field;
  *target = beta == 0.0 ? value : value + beta * *target;
}

__global__ void butterfly_projection_reverse_kernel(
    const double *__restrict__ input, double *__restrict__ output,
    int leading_dimension, const int *__restrict__ ranks,
    const int *__restrict__ columns, const std::int64_t *__restrict__ src,
    const std::int64_t *__restrict__ dst,
    const std::int64_t *__restrict__ factor_offsets,
    const std::int64_t *__restrict__ pivot_offsets,
    const int *__restrict__ tile_groups,
    const int *__restrict__ tile_starts,
    const int *__restrict__ pivots, const double *__restrict__ factors,
    std::int64_t add_src, std::int64_t add_dst, double beta) {
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const int group = tile_groups[blockIdx.x];
  const int column = tile_starts[blockIdx.x] + warp;
  const int field = static_cast<int>(blockIdx.y) * warp_size + lane;
  const int column_count = columns[group];
  if (column >= column_count || field >= leading_dimension)
    return;

  const int rank = ranks[group];
  const std::int64_t factor_base = factor_offsets[group];
  double value;
  if (column < rank) {
    value =
        input[(src[group] + add_src + column) * leading_dimension + field];
  } else {
    value = 0.0;
    for (int row = 0; row < rank; ++row) {
      const double factor =
          factors[factor_base + static_cast<std::int64_t>(column - rank) *
                                    rank +
                  row];
      value = fma(input[(src[group] + add_src + row) * leading_dimension +
                        field],
                  factor, value);
    }
  }

  const std::int64_t pivot_base = pivot_offsets[group];
  double *target =
      output +
      (dst[group] + add_dst + pivots[pivot_base + column]) *
          leading_dimension +
      field;
  *target = beta == 0.0 ? value : value + beta * *target;
}

} // namespace

extern "C" void ectrans_butterfly_projection_cuda(
    const double *input, double *output, int leading_dimension,
    const int *ranks, const int *columns, const std::int64_t *src,
    const std::int64_t *dst, const std::int64_t *factor_offsets,
    const std::int64_t *pivot_offsets, const int *tile_groups,
    const int *tile_starts, const int *pivots, const double *factors,
    std::int64_t add_src, std::int64_t add_dst, int tile_count, int reverse,
    double beta, std::intptr_t stream_value) {
  if (leading_dimension <= 0 || tile_count <= 0)
    return;

  const dim3 block(warp_size * warps_per_block);
  const dim3 grid(static_cast<unsigned int>(tile_count),
                  static_cast<unsigned int>((leading_dimension + warp_size -
                                             1) /
                                            warp_size));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  if (reverse) {
    butterfly_projection_reverse_kernel<<<grid, block, 0, stream>>>(
        input, output, leading_dimension, ranks, columns, src, dst,
        factor_offsets, pivot_offsets, tile_groups, tile_starts, pivots,
        factors, add_src, add_dst, beta);
  } else {
    butterfly_projection_forward_kernel<<<grid, block, 0, stream>>>(
        input, output, leading_dimension, ranks, columns, src, dst,
        factor_offsets, pivot_offsets, tile_groups, tile_starts, pivots,
        factors, add_src, add_dst, beta);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "butterfly projection CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

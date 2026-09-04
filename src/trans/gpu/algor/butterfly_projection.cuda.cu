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

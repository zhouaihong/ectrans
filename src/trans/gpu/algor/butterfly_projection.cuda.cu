// (C) Copyright 2026- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int fields_per_block = 16;
constexpr int outputs_per_block = 8;
constexpr int reduction_tile = 8;

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
  __shared__ double input_tile[reduction_tile][fields_per_block];
  __shared__ double factor_tile[reduction_tile][outputs_per_block];

  const int field_lane = threadIdx.x;
  const int output_lane = threadIdx.y;
  const int thread = output_lane * fields_per_block + field_lane;
  const int group = tile_groups[blockIdx.x];
  const int output_column = tile_starts[blockIdx.x] + output_lane;
  const int field = static_cast<int>(blockIdx.y) * fields_per_block + field_lane;
  const int rank = ranks[group];
  const int column_count = columns[group];
  const std::int64_t pivot_base = pivot_offsets[group];
  const std::int64_t factor_base = factor_offsets[group];
  const bool active = output_column < rank && field < leading_dimension;
  double value = active
                     ? input[(src[group] + add_src +
                              pivots[pivot_base + output_column]) *
                                 leading_dimension +
                             field]
                     : 0.0;

  for (int column_base = rank; column_base < column_count;
       column_base += reduction_tile) {
    const int count = min(reduction_tile, column_count - column_base);
    for (int index = thread; index < count * fields_per_block;
         index += fields_per_block * outputs_per_block) {
      const int column = column_base + index / fields_per_block;
      const int tile_field = index % fields_per_block;
      const int global_field =
          static_cast<int>(blockIdx.y) * fields_per_block + tile_field;
      input_tile[index / fields_per_block][tile_field] =
          global_field < leading_dimension
              ? input[(src[group] + add_src + pivots[pivot_base + column]) *
                          leading_dimension +
                      global_field]
              : 0.0;
    }
    for (int index = thread; index < count * outputs_per_block;
         index += fields_per_block * outputs_per_block) {
      const int column = column_base + index / outputs_per_block;
      const int tile_output = index % outputs_per_block;
      const int global_output = tile_starts[blockIdx.x] + tile_output;
      factor_tile[index / outputs_per_block][tile_output] =
          global_output < rank
              ? factors[factor_base +
                        static_cast<std::int64_t>(column - rank) * rank +
                        global_output]
              : 0.0;
    }
    __syncthreads();
    if (active)
      for (int column = 0; column < count; ++column)
        value = fma(input_tile[column][field_lane],
                    factor_tile[column][output_lane], value);
    __syncthreads();
  }

  if (active) {
    double *target =
        output + (dst[group] + add_dst + output_column) * leading_dimension +
        field;
    *target = beta == 0.0 ? value : value + beta * *target;
  }
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
  __shared__ double input_tile[reduction_tile][fields_per_block];
  __shared__ double factor_tile[reduction_tile][outputs_per_block];

  const int field_lane = threadIdx.x;
  const int output_lane = threadIdx.y;
  const int thread = output_lane * fields_per_block + field_lane;
  const int group = tile_groups[blockIdx.x];
  const int column = tile_starts[blockIdx.x] + output_lane;
  const int field = static_cast<int>(blockIdx.y) * fields_per_block + field_lane;
  const int column_count = columns[group];
  const int rank = ranks[group];
  const std::int64_t factor_base = factor_offsets[group];
  const bool active = column < column_count && field < leading_dimension;
  double value = active && column < rank
                     ? input[(dst[group] + add_src + column) * leading_dimension +
                             field]
                     : 0.0;

  for (int row_base = 0; row_base < rank; row_base += reduction_tile) {
    const int count = min(reduction_tile, rank - row_base);
    for (int index = thread; index < count * fields_per_block;
         index += fields_per_block * outputs_per_block) {
      const int row = row_base + index / fields_per_block;
      const int tile_field = index % fields_per_block;
      const int global_field =
          static_cast<int>(blockIdx.y) * fields_per_block + tile_field;
      input_tile[index / fields_per_block][tile_field] =
          global_field < leading_dimension
              ? input[(dst[group] + add_src + row) * leading_dimension +
                      global_field]
              : 0.0;
    }
    for (int index = thread; index < count * outputs_per_block;
         index += fields_per_block * outputs_per_block) {
      const int row = row_base + index / outputs_per_block;
      const int tile_output = index % outputs_per_block;
      const int global_column = tile_starts[blockIdx.x] + tile_output;
      factor_tile[index / outputs_per_block][tile_output] =
          global_column >= rank && global_column < column_count
              ? factors[factor_base +
                        static_cast<std::int64_t>(global_column - rank) * rank +
                        row]
              : 0.0;
    }
    __syncthreads();
    if (active && column >= rank)
      for (int row = 0; row < count; ++row)
        value = fma(input_tile[row][field_lane],
                    factor_tile[row][output_lane], value);
    __syncthreads();
  }

  if (active) {
    const std::int64_t pivot_base = pivot_offsets[group];
    double *target =
        output +
        (src[group] + add_dst + pivots[pivot_base + column]) *
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

  const dim3 block(fields_per_block, outputs_per_block);
  const dim3 grid(static_cast<unsigned int>(tile_count),
                  static_cast<unsigned int>((leading_dimension +
                                             fields_per_block - 1) /
                                            fields_per_block));
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

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
constexpr int warps_per_block = 8;
constexpr int degree_tile = 32;

__global__ void legendre_recurrence_inverse_kernel(
    const double *__restrict__ matrix, const double *__restrict__ input,
    double *__restrict__ output, int leading_dimension, int fields, int nsmax,
    int symmetric, const int *__restrict__ ms,
    const std::int64_t *__restrict__ row_offsets,
    const std::int64_t *__restrict__ coefficient_offsets,
    const std::int64_t *__restrict__ matrix_offsets,
    const int *__restrict__ matrix_strides,
    const int *__restrict__ tile_m_indices,
    const std::int64_t *__restrict__ tile_rows,
    const double *__restrict__ coefficient1,
    const double *__restrict__ coefficient2,
    const double *__restrict__ coefficient3,
    const double *__restrict__ mu_squared, const int *__restrict__ seeds) {
  __shared__ double input_tile[degree_tile][warp_size];
  const int thread = threadIdx.x;
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const int m_index = tile_m_indices[blockIdx.x] - 1;
  const std::int64_t row = tile_rows[blockIdx.x] + warp;
  const int field = static_cast<int>(blockIdx.y) * warp_size + lane;
  const int m = ms[m_index];
  const bool valid_row = row < row_offsets[m_index + 1];

  const int columns = (nsmax - m + (symmetric ? 3 : 2)) / 2;
  const int seed = valid_row ? seeds[row] : 0;
  const std::int64_t local_row = row - row_offsets[m_index];
  const std::int64_t input_base =
      coefficient_offsets[m_index] * leading_dimension;
  const std::int64_t matrix_base = matrix_offsets[m_index];
  const std::int64_t coefficient_base = coefficient_offsets[m_index];
  const int matrix_stride = matrix_strides[m_index];

  double sum = 0.0;
  double previous2 = 0.0;
  double previous1 = 0.0;
  const double mu2 = valid_row ? mu_squared[row] : 0.0;
  for (int column_top = columns; column_top >= 1;
       column_top -= degree_tile) {
    for (int position = thread; position < degree_tile * warp_size;
         position += warp_size * warps_per_block) {
      const int degree = position / warp_size;
      const int field_in_tile = position & (warp_size - 1);
      const int input_field =
          static_cast<int>(blockIdx.y) * warp_size + field_in_tile;
      const int column = column_top - degree;
      input_tile[degree][field_in_tile] =
          column >= 1 && input_field < fields
              ? input[input_base +
                      static_cast<std::int64_t>(column - 1) *
                          leading_dimension +
                      input_field]
              : 0.0;
    }
    __syncthreads();

    if (valid_row) {
      for (int degree = 0; degree < degree_tile; ++degree) {
        const int column = column_top - degree;
        if (column < 1)
          break;
        double polynomial = 0.0;
        if (lane == 0) {
          const std::int64_t matrix_position =
              matrix_base +
              static_cast<std::int64_t>(column - 1) * matrix_stride +
              local_row;
          if (seed < 2 || column > seed) {
            polynomial = matrix[matrix_position];
          } else if (column == seed) {
            previous2 = matrix[matrix_position];
            polynomial = previous2;
          } else if (column == seed - 1) {
            previous1 = matrix[matrix_position];
            polynomial = previous1;
          } else {
            const std::int64_t coefficient_position =
                coefficient_base + column - 1;
            polynomial =
                (coefficient1[coefficient_position] * mu2 +
                 coefficient2[coefficient_position]) *
                    previous1 +
                coefficient3[coefficient_position] * previous2;
            previous2 = previous1;
            previous1 = polynomial;
          }
        }
        polynomial = __shfl_sync(0xffffffffu, polynomial, 0);
        if (field < fields)
          sum = fma(input_tile[degree][lane], polynomial, sum);
      }
    }
    __syncthreads();
  }

  if (valid_row && field < fields)
    output[row * leading_dimension + field] = sum;
}

} // namespace

extern "C" void ectrans_legendre_recurrence_inverse_cuda(
    const double *matrix, const double *input, double *output,
    int leading_dimension, int fields, int nsmax, int symmetric,
    const int *ms, const std::int64_t *row_offsets,
    const std::int64_t *coefficient_offsets,
    const std::int64_t *matrix_offsets, const int *matrix_strides,
    const int *tile_m_indices, const std::int64_t *tile_rows,
    const double *coefficient1, const double *coefficient2,
    const double *coefficient3, const double *mu_squared, const int *seeds,
    int tile_count, std::intptr_t stream_value) {
  if (fields <= 0 || tile_count <= 0)
    return;
  const dim3 block(warp_size * warps_per_block);
  const dim3 grid(static_cast<unsigned int>(tile_count),
                  static_cast<unsigned int>((fields + warp_size - 1) /
                                            warp_size));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  legendre_recurrence_inverse_kernel<<<grid, block, 0, stream>>>(
      matrix, input, output, leading_dimension, fields, nsmax, symmetric, ms,
      row_offsets, coefficient_offsets, matrix_offsets, matrix_strides,
      tile_m_indices, tile_rows, coefficient1, coefficient2, coefficient3,
      mu_squared, seeds);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "Legendre recurrence CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

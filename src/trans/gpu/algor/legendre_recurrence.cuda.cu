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

__device__ __forceinline__ int find_m_index(std::int64_t row,
                                             const std::int64_t *offsets,
                                             int count) {
  int lower = 0;
  int upper = count;
  while (lower + 1 < upper) {
    const int middle = lower + (upper - lower) / 2;
    if (offsets[middle] <= row)
      lower = middle;
    else
      upper = middle;
  }
  return lower;
}

__global__ void legendre_recurrence_inverse_kernel(
    const double *__restrict__ matrix, const double *__restrict__ input,
    double *__restrict__ output, int leading_dimension, int fields, int nsmax,
    int m_count, int symmetric, const int *__restrict__ ms,
    const std::int64_t *__restrict__ row_offsets,
    const std::int64_t *__restrict__ coefficient_offsets,
    const std::int64_t *__restrict__ matrix_offsets,
    const int *__restrict__ matrix_strides,
    const double *__restrict__ coefficient1,
    const double *__restrict__ coefficient2,
    const double *__restrict__ coefficient3,
    const double *__restrict__ mu_squared, const int *__restrict__ seeds,
    std::int64_t total_rows) {
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const std::int64_t row =
      static_cast<std::int64_t>(blockIdx.x) * warps_per_block + warp;
  const int field = static_cast<int>(blockIdx.y) * warp_size + lane;
  if (row >= total_rows)
    return;

  const int m_index = find_m_index(row, row_offsets, m_count);
  const int m = ms[m_index];
  if (m == 0)
    return;

  const int columns = (nsmax - m + (symmetric ? 3 : 2)) / 2;
  const int seed = seeds[row];
  const std::int64_t local_row = row - row_offsets[m_index];
  const std::int64_t input_base =
      coefficient_offsets[m_index] * leading_dimension;
  const std::int64_t matrix_base = matrix_offsets[m_index];
  const std::int64_t coefficient_base = coefficient_offsets[m_index];
  const int matrix_stride = matrix_strides[m_index];

  double sum = 0.0;
  double previous2 = 0.0;
  double previous1 = 0.0;
  const double mu2 = mu_squared[row];
  for (int column = columns; column >= 1; --column) {
    double polynomial = 0.0;
    if (lane == 0) {
      const std::int64_t matrix_position =
          matrix_base + static_cast<std::int64_t>(column - 1) * matrix_stride +
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
    if (field < fields) {
      const std::int64_t input_position =
          input_base + static_cast<std::int64_t>(column - 1) *
                           leading_dimension +
          field;
      sum = fma(input[input_position], polynomial, sum);
    }
  }

  if (field < fields)
    output[row * leading_dimension + field] = sum;
}

} // namespace

extern "C" void ectrans_legendre_recurrence_inverse_cuda(
    const double *matrix, const double *input, double *output,
    int leading_dimension, int fields, int nsmax, int m_count, int symmetric,
    const int *ms, const std::int64_t *row_offsets,
    const std::int64_t *coefficient_offsets,
    const std::int64_t *matrix_offsets, const int *matrix_strides,
    const double *coefficient1, const double *coefficient2,
    const double *coefficient3, const double *mu_squared, const int *seeds,
    std::int64_t total_rows, std::intptr_t stream_value) {
  if (fields <= 0 || total_rows <= 0)
    return;
  const dim3 block(warp_size * warps_per_block);
  const dim3 grid(static_cast<unsigned int>(
                      (total_rows + warps_per_block - 1) / warps_per_block),
                  static_cast<unsigned int>((fields + warp_size - 1) /
                                            warp_size));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  legendre_recurrence_inverse_kernel<<<grid, block, 0, stream>>>(
      matrix, input, output, leading_dimension, fields, nsmax, m_count,
      symmetric, ms, row_offsets, coefficient_offsets, matrix_offsets,
      matrix_strides, coefficient1, coefficient2, coefficient3, mu_squared,
      seeds, total_rows);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "Legendre recurrence CUDA launch failed: %s\n",
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

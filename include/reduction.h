#pragma once

#include "common.h"
#include <cuda_runtime.h>

/**
 * GPU kernel: reduces 'n' elements from 'input' to one max per block.
 *
 * Each block handles 2 × blockDim.x input elements (grid-stride load),
 * then performs a binary-tree reduction in shared memory, followed by a
 * warp-level shuffle reduction for the last 32 elements.
 *
 * @param input   Device input array  (length n).
 * @param output  Device output array (length == number of blocks launched).
 * @param n       Number of elements to reduce.
 *
 * Shared memory required: blockDim.x * sizeof(double).
 */
__global__ void reduce_max_kernel(
    const double* __restrict__ input,
    double* __restrict__       output,
    int n
);

/**
 * Host wrapper: computes max over d_data[0 .. n-1] on the GPU.
 * Allocates and frees temporary device memory internally.
 * Calls cudaDeviceSynchronize() before returning.
 *
 * @param d_data  Device array (unchanged on return).
 * @param n       Number of elements.
 * @return        Maximum value found.
 */
double reduce_max_gpu(const double* d_data, int n);
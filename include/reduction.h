/**
 * reduction.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Stand-alone parallel max-reduction on a flat GPU array.
 *
 * Algorithm (two-pass):
 *   1. reduce_max_kernel  –  each block reduces 2 × blockDim.x elements
 *      to one value using shared memory (binary tree) + warp shuffle for
 *      the final 32 elements (__shfl_down_sync, no __syncthreads needed).
 *   2. If more than one block was launched, a second kernel call reduces
 *      the partial results to a single scalar, which is copied to the host.
 *
 * Usage:
 *   double max_val = reduce_max_gpu(d_array, n);
 * ─────────────────────────────────────────────────────────────────────────────
 */

#pragma once

#include <cuda_runtime.h>
#include "common.h"

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
    int n);

/**
 * Host wrapper: computes max over d_data[0..n-1] on the GPU.
 * Allocates and frees temporary device memory internally.
 * Calls cudaDeviceSynchronize() before returning.
 *
 * @param d_data  Device array (unchanged on return).
 * @param n       Number of elements.
 * @return        Maximum value found.
 */
double reduce_max_gpu(const double* d_data, int n);

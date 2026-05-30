/**
 * reduction.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * Stand-alone parallel max-reduction over a flat device array.
 *
 * Strategy
 * ────────
 *   Phase 1 – reduce_max_kernel
 *     Each block reads 2 × blockDim.x consecutive elements (grid-stride load)
 *     and reduces them to a single block-max in shared memory using:
 *       a) Binary-tree reduction for strides 32 < s ≤ blockDim.x/2.
 *       b) Warp-shuffle reduction (__shfl_down_sync) for the final 32 threads,
 *          eliminating 5 × __syncthreads() and 5 extra shared-memory rounds.
 *
 *   Phase 2 – if gridDim.x > 1, run reduce_max_kernel again on partial results.
 *
 * Result is fetched to the host with a single cudaMemcpy.
 *
 * Note: this standalone reducer is used in main.cu for final validation.
 *       The Jacobi V2 kernel embeds its own per-block reduction (see gpu_solver.cu).
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cfloat>
#include <cuda_runtime.h>

#include "../include/reduction.h"

// ─────────────────────────────────────────────────────────────────────────────
// Device helper
// No __syncthreads() needed — all threads belong to the same warp.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
double warpReduceMax(double val)
{
    // Full warp mask — all 32 lanes participate.
    unsigned int mask = 0xffffffff;

    // 5 rounds of butterfly reduction (log₂ 32 = 5).
    val = fmax(val, __shfl_down_sync(mask, val, 16));
    val = fmax(val, __shfl_down_sync(mask, val,  8));
    val = fmax(val, __shfl_down_sync(mask, val,  4));
    val = fmax(val, __shfl_down_sync(mask, val,  2));
    val = fmax(val, __shfl_down_sync(mask, val,  1));
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: blockDim.x must be a power of 2 and >= 64.
// Shared memory required: blockDim.x * sizeof(double).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduce_max_kernel(
    const double* __restrict__ input,
    double* __restrict__       output,
    int n
) {
    extern __shared__ double sdata[];

    const int tid = threadIdx.x;

    // ── Grid-stride load: each thread handles two consecutive elements
    // This halves the number of blocks needed and keeps threads busy longer.
    int i              = (int)(blockIdx.x * (blockDim.x * 2) + tid);
    const int gridSize = (int)(blockDim.x * gridDim.x * 2);
    double myMax       = -DBL_MAX;

    while (i < n) {
        myMax = fmax(myMax, input[i]);
        if (i + blockDim.x < n) myMax = fmax(myMax, input[i + blockDim.x]);
        i += gridSize;
    }

    sdata[tid] = myMax;
    __syncthreads();

    // ── Binary-tree reduction (shared memory) — strides > warp size
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // ── Warp-level reduction — no __syncthreads() needed
    // After the loop above, sdata[0 .. 63] have partial results.
    // Threads 0-31 pick up the values from 32-63 and reduce within the warp.
    if (tid < 32) {
        // Bring in the second half of the remaining 64 elements.
        double v = fmax(sdata[tid], sdata[tid + 32]);

        // Warp shuffle reduction (no shared-memory write needed).
        v = warpReduceMax(v);
        if (tid == 0) output[blockIdx.x] = v;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrapper
// ─────────────────────────────────────────────────────────────────────────────
double reduce_max_gpu(const double* d_data, int n) {
    if (n <= 0) return 0.0;

    const int threads = 256;   // must be a power of 2 and >= 64

    // Each block covers 2 × threads elements (grid-stride load).
    const int blocks  = (n + threads * 2 - 1) / (threads * 2);

    double* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)blocks * sizeof(double)));

    // Phase 1: reduce n elements → blocks partial maxima.
    reduce_max_kernel<<<blocks, threads, (size_t)threads * sizeof(double)>>>(d_data, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    double result = 0.0;

    if (blocks > 1) {
        // Phase 2: reduce partial maxima → single value.
        double* d_final = nullptr;
        CUDA_CHECK(cudaMalloc(&d_final, sizeof(double)));

        reduce_max_kernel<<<1, threads, (size_t)threads * sizeof(double)>>>(d_partial, d_final, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&result, d_final, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_final));
    } else {
        // Only one block: result is already in d_partial[0].
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_partial));
    return result;
}
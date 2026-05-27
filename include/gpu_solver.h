/**
 * gpu_solver.h
 * ─────────────────────────────────────────────────────────────────────────────
 * GPU Jacobi kernels and host-side solver wrappers.
 *
 * Two kernel variants are provided:
 *
 *   V1 – jacobi_kernel_naive
 *        One thread per grid point; all reads/writes go to global memory.
 *        Convergence check (Strategy A): every check_every iterations the
 *        two grids are copied to the host and the error is computed on CPU.
 *
 *   V2 – jacobi_kernel_shared
 *        Shared-memory tile with 1-cell halo; convergence check embedded
 *        (Strategy B): each block computes its local max-diff and writes it
 *        to d_block_max; the host reads the array every check_every iters.
 *
 * CUDA design notes
 * ─────────────────
 *   • threadIdx.x ↔ column j   (fast/contiguous dimension → coalesced loads)
 *   • threadIdx.y ↔ row    i
 *   • __restrict__ hints the compiler that pointers do not alias.
 *   • __syncthreads() used after every shared-memory load phase.
 *   • d_block_max uses double atomics implicitly via the binary-tree
 *     reduction inside the kernel (no atomicMax needed).
 * ─────────────────────────────────────────────────────────────────────────────
 */

#pragma once

#include <cuda_runtime.h>
#include "common.h"

// ── Kernel declarations ───────────────────────────────────────────────────────

/**
 * V1 – Naïve Jacobi kernel (global memory only).
 *
 * Each thread (tx, ty) processes grid point (i = blockIdx.y*TILE_Y+ty,
 * j = blockIdx.x*TILE_X+tx).  Interior-only guard skips boundary threads.
 */
__global__ void jacobi_kernel_naive(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2);

/**
 * V2 – Optimised Jacobi kernel (shared memory + embedded max reduction).
 *
 * Shared memory layout:
 *   s_u   [SMEM_Y][SMEM_X]  – tile of u with 1-cell halo (SMEM_* = TILE_*+2)
 *   s_max [TILE_Y * TILE_X] – per-thread partial max for block reduction
 *
 * After __syncthreads() the Jacobi update reads exclusively from s_u,
 * saving 4 redundant global loads per interior point (vs. V1).
 *
 * @param d_block_max  Output array of length gridDim.x * gridDim.y;
 *                     thread 0 of each block writes the block's max |diff|.
 */
__global__ void jacobi_kernel_shared(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2,
    double* __restrict__       d_block_max);

/**
 * V3 – Super-optimised Jacobi kernel (linear, coalesced shared-memory loader + embedded max reduction).
 *
 * All 256 threads of the block cooperate to load the 324 elements of the SMEM tile
 * (16x16 center + 1-cell halo) in exactly 2 coalesced read phases.
 */
__global__ void jacobi_kernel_shared_coalesced(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2,
    double* __restrict__       d_block_max);

// ── Host-side solver wrappers ─────────────────────────────────────────────────

/**
 * Run V1 on already-allocated, already-initialised device arrays.
 * The caller owns d_u and d_u_new; on return d_u holds the final solution.
 */
SolverResult jacobi_gpu_naive(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f);

/**
 * Run V2 on already-allocated, already-initialised device arrays.
 * The caller owns d_u and d_u_new; on return d_u holds the final solution.
 */
SolverResult jacobi_gpu_optimized(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f);

/**
 * Run V3 (coalesced shmem loading) on already-allocated, already-initialised device arrays.
 * The caller owns d_u and d_u_new; on return d_u holds the final solution.
 */
SolverResult jacobi_gpu_coalesced(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f);

/**
 * Pure-throughput benchmark: runs exactly bench_iters iterations of the
 * chosen kernel without any convergence check or H↔D transfer.
 * Returns milliseconds for bench_iters iterations (measured with CUDA events).
 *
 * @param use_shared  0 → V1 kernel; 1 → V2 kernel; 2 → V3 kernel.
 */
float jacobi_gpu_benchmark(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f,
    int           bench_iters,
    int           version);


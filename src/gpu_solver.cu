/**
 * gpu_solver.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * CUDA implementation of the Jacobi solver for −Δu = f on [0,1]².
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ V1 – jacobi_kernel_naive                                                │
 * │   • One thread per grid point, pure global memory.                      │
 * │   • threadIdx.x → column j  →  coalesced loads/stores.                 │
 * │   • Convergence check (Strategy A): every check_every iterations both  │
 * │     grids are copied to the host and max|diff| is computed on CPU.     │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ V2 – jacobi_kernel_shared                                               │
 * │   • Shared-memory tile (TILE_Y+2)×(TILE_X+2) with 1-cell halo.        │
 * │   • threadIdx.x → column j  →  coalesced global loads.                 │
 * │   • __syncthreads() after tile+halo load; updates read only shmem.     │
 * │   • Embedded binary-tree max-reduction for convergence (Strategy B):   │
 * │     each block writes its max diff to d_block_max[blockIdx.y*gdx+gdx]. │
 * │   • Host reads d_block_max every check_every iterations.               │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Coalescence analysis
 * ─────────────────────
 *   Row-major layout: u[i,j] = u[i*N+j].  A warp spans 32 consecutive
 *   threadIdx.x values, so threads in a warp access u[i, j..j+31] which
 *   are 32 consecutive doubles → 256-byte aligned, fully coalesced.
 *
 *   Centre tile loads: one load per thread, all 16 threads in a half-warp
 *   (for TILE_X=16) access u[i, blockX*16 .. blockX*16+15] → coalesced. ✓
 *   Top/bottom halo:  TILE_X threads load one consecutive row → coalesced. ✓
 *   Left/right halo:  TILE_Y threads load column-wise (stride N) → strided,
 *   but only 16 such loads per block — minor overhead. ✓
 *
 * Shared memory layout (double, row-major)
 * ─────────────────────────────────────────
 *   s_u  : (TILE_Y+2) × (TILE_X+2)  doubles = SMEM_Y × SMEM_X doubles
 *   s_max: TILE_Y × TILE_X           doubles  (reused after computation)
 *
 *   s_u[si][sj]  where  si = threadIdx.y+1,  sj = threadIdx.x+1  (halo offset)
 *
 *   Corner cells s_u[0][0], s_u[0][SMEM_X-1], etc. are allocated but never
 *   accessed (5-point stencil uses only cardinal neighbours). ✓
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

#include "../include/gpu_solver.h"
#include "../include/timer.h"

// ═════════════════════════════════════════════════════════════════════════════
// V1 — Naïve kernel (global memory only)
// ═════════════════════════════════════════════════════════════════════════════
__global__ void jacobi_kernel_naive(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2)
{
    // Map thread indices: x → column j (fast, coalesced), y → row i.
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    // Skip boundary and out-of-range threads.
    if (i < 1 || i > N - 2 || j < 1 || j > N - 2) return;

    const int id = IDX(i, j, N);

    // 5-point Jacobi update — reads 4 neighbours + f from global memory.
    u_new[id] = 0.25 * (
        u[IDX(i - 1, j, N)] +   // top    (stride  N doubles → not coalesced,
        u[IDX(i + 1, j, N)] +   // bottom    but different threads → interleaved)
        u[IDX(i, j - 1, N)] +   // left   (j-1: contiguous-1 → near-coalesced)
        u[IDX(i, j + 1, N)] +   // right  (j+1: contiguous+1 → near-coalesced)
        h2 * f[id]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// V2 — Optimised kernel: shared-memory tile + halo + embedded max-reduction
// ═════════════════════════════════════════════════════════════════════════════
__global__ void jacobi_kernel_shared(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2,
    double* __restrict__       d_block_max)
{
    // ── Shared memory ─────────────────────────────────────────────────────────
    // s_u  : tile of u with 1-cell halo on every side.
    //         Row-major inside shared memory: row width = SMEM_X = TILE_X+2.
    // s_max: per-thread scratch for the block-level max-reduction.
    //         Declared separately to keep the two phases clearly distinct.
    __shared__ double s_u  [SMEM_Y * SMEM_X];   // (TILE_Y+2) × (TILE_X+2)
    __shared__ double s_max[TILE_Y  * TILE_X];   // TILE_Y × TILE_X

    // ── Thread / global indices ────────────────────────────────────────────────
    const int tx = (int)threadIdx.x;   // 0 .. TILE_X-1  (column direction)
    const int ty = (int)threadIdx.y;   // 0 .. TILE_Y-1  (row    direction)

    // Global grid position of this thread's interior point.
    const int gj = (int)(blockIdx.x * TILE_X + tx);   // column
    const int gi = (int)(blockIdx.y * TILE_Y + ty);   // row

    // Corresponding position inside the shared-memory tile (halo offset = +1).
    const int sj = tx + 1;
    const int si = ty + 1;

    // ── Phase 1: Load tile centre into shared memory ───────────────────────────
    // Each thread loads exactly one element.  Boundary / out-of-grid → clamp to 0
    // (consistent with Dirichlet u = 0 on ∂Ω).
    double u_center;
    if (gi < N && gj < N)
        u_center = u[IDX(gi, gj, N)];
    else
        u_center = 0.0;

    s_u[si * SMEM_X + sj] = u_center;

    // ── Phase 2: Load halo rows / columns ─────────────────────────────────────
    // Only edge threads of the block perform the extra loads.
    // Top row (gi - 1): TILE_X threads → coalesced (consecutive columns).
    if (ty == 0) {
        const int hi = gi - 1;
        s_u[0 * SMEM_X + sj] =
            (hi >= 0 && gj < N) ? u[IDX(hi, gj, N)] : 0.0;
    }
    // Bottom row (gi + 1): TILE_X threads → coalesced.
    if (ty == TILE_Y - 1) {
        const int hi = gi + 1;
        s_u[(TILE_Y + 1) * SMEM_X + sj] =
            (hi < N && gj < N) ? u[IDX(hi, gj, N)] : 0.0;
    }
    // Left column (gj - 1): TILE_Y threads → strided (stride N), minor overhead.
    if (tx == 0) {
        const int hj = gj - 1;
        s_u[si * SMEM_X + 0] =
            (gi < N && hj >= 0) ? u[IDX(gi, hj, N)] : 0.0;
    }
    // Right column (gj + 1): TILE_Y threads → strided, minor overhead.
    if (tx == TILE_X - 1) {
        const int hj = gj + 1;
        s_u[si * SMEM_X + (TILE_X + 1)] =
            (gi < N && hj < N) ? u[IDX(gi, hj, N)] : 0.0;
    }

    // ── Barrier: all threads must finish loading before any thread computes ────
    __syncthreads();

    // ── Phase 3: Compute Jacobi update (reads only from shared memory) ─────────
    double diff = 0.0;

    if (gi >= 1 && gi <= N - 2 && gj >= 1 && gj <= N - 2) {
        const int    id  = IDX(gi, gj, N);

        // All four reads hit L1/shared memory — no global traffic.
        const double val = 0.25 * (
            s_u[(si - 1) * SMEM_X + sj ] +   // top
            s_u[(si + 1) * SMEM_X + sj ] +   // bottom
            s_u[ si      * SMEM_X + sj - 1] + // left
            s_u[ si      * SMEM_X + sj + 1] + // right
            h2 * f[id]                          // f still from global (read once)
        );
        u_new[id] = val;

        // Local contribution to the block's convergence error.
        // u_center is the old value already in a register — no extra read.
        diff = fabs(val - u_center);
    }

    // ── Phase 4: Block-level parallel max-reduction (binary tree) ─────────────
    // Each thread writes its local diff into s_max[tid].
    const int tid = ty * TILE_X + tx;
    s_max[tid] = diff;

    // Barrier: all diffs must be written before reduction starts.
    __syncthreads();

    // Binary tree: halve the number of active threads at every step.
    for (int stride = (TILE_X * TILE_Y) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other = s_max[tid + stride];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        // Must sync before the next half reads from s_max.
        __syncthreads();
    }

    // Thread 0 writes the block maximum to global memory.
    // Index: blocks are laid out row-major in the grid.
    if (tid == 0) {
        const int block_id = (int)(blockIdx.y * gridDim.x + blockIdx.x);
        d_block_max[block_id] = s_max[0];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// V3 — Super-optimised kernel: coalesced shared-memory tile loader + block reduction
// ═════════════════════════════════════════════════════════════════════════════
__global__ void jacobi_kernel_shared_coalesced(
    double* __restrict__       u_new,
    const double* __restrict__ u,
    const double* __restrict__ f,
    int    N,
    double h2,
    double* __restrict__       d_block_max)
{
    // s_u  : tile of u with 1-cell halo. Row width = SMEM_X = TILE_X + 2.
    // s_max: per-thread scratch for block-level max-reduction.
    __shared__ double s_u  [SMEM_Y * SMEM_X];   // 18 × 18 = 324 elements
    __shared__ double s_max[TILE_Y  * TILE_X];   // 16 × 16 = 256 elements

    const int tx  = (int)threadIdx.x;
    const int ty  = (int)threadIdx.y;
    const int tid = ty * TILE_X + tx;   // 0 .. 255

    // ── Coalesced Linear Loading ─────────────────────────────────────────────
    // We have 256 threads loading 324 elements.
    // 1st step: load elements 0..255
    {
        const int linear_id = tid;
        const int li = linear_id / SMEM_X;
        const int lj = linear_id % SMEM_X;
        const int gi = (int)blockIdx.y * TILE_Y + li - 1;
        const int gj = (int)blockIdx.x * TILE_X + lj - 1;

        if (gi >= 0 && gi < N && gj >= 0 && gj < N) {
            s_u[linear_id] = u[IDX(gi, gj, N)];
        } else {
            s_u[linear_id] = 0.0;
        }
    }
    // 2nd step: load remaining 68 elements (256..323)
    if (tid < 68) {
        const int linear_id = tid + 256;
        const int li = linear_id / SMEM_X;
        const int lj = linear_id % SMEM_X;
        const int gi = (int)blockIdx.y * TILE_Y + li - 1;
        const int gj = (int)blockIdx.x * TILE_X + lj - 1;

        if (gi >= 0 && gi < N && gj >= 0 && gj < N) {
            s_u[linear_id] = u[IDX(gi, gj, N)];
        } else {
            s_u[linear_id] = 0.0;
        }
    }

    // Ensure all threads finished loading to shared memory
    __syncthreads();

    // ── Phase 3: Jacobi Update ──
    const int gj = (int)(blockIdx.x * TILE_X + tx);
    const int gi = (int)(blockIdx.y * TILE_Y + ty);
    const int sj = tx + 1;
    const int si = ty + 1;

    double diff = 0.0;
    const double u_center = s_u[si * SMEM_X + sj];

    if (gi >= 1 && gi <= N - 2 && gj >= 1 && gj <= N - 2) {
        const int id = IDX(gi, gj, N);
        const double val = 0.25 * (
            s_u[(si - 1) * SMEM_X + sj ] +   // top
            s_u[(si + 1) * SMEM_X + sj ] +   // bottom
            s_u[ si      * SMEM_X + sj - 1] + // left
            s_u[ si      * SMEM_X + sj + 1] + // right
            h2 * f[id]
        );
        u_new[id] = val;

        diff = fabs(val - u_center);
    }

    // ── Phase 4: Block reduction ──
    s_max[tid] = diff;
    __syncthreads();

    for (int stride = (TILE_X * TILE_Y) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other = s_max[tid + stride];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int block_id = (int)(blockIdx.y * gridDim.x + blockIdx.x);
        d_block_max[block_id] = s_max[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────

// Internal helper: CPU-side reduction of the per-block max array.
// Called after cudaMemcpy of d_block_max to the host.
// ─────────────────────────────────────────────────────────────────────────────
static double host_reduce_max(const double* h_arr, int n)
{
    double mx = 0.0;
    for (int b = 0; b < n; ++b)
        if (h_arr[b] > mx) mx = h_arr[b];
    return mx;
}

// ═════════════════════════════════════════════════════════════════════════════
// V1 Host-side solver wrapper
// ═════════════════════════════════════════════════════════════════════════════
SolverResult jacobi_gpu_naive(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f)
{
    const int    N           = params.N;
    const double h2          = params.h * params.h;
    const int    maxi        = params.max_iter;
    const double tol         = params.tol;
    const int    check_every = params.check_every;

    // Grid covers the full N×N array (interior guard is inside the kernel).
    dim3 block(TILE_X, TILE_Y);
    dim3 grid(
        (unsigned int)((N + TILE_X - 1) / TILE_X),
        (unsigned int)((N + TILE_Y - 1) / TILE_Y)
    );

    // Host buffers for Strategy-A convergence check (copy entire grids).
    const size_t bytes = (size_t)N * N * sizeof(double);
    double* h_u     = (double*)malloc(bytes);
    double* h_u_new = (double*)malloc(bytes);
    if (!h_u || !h_u_new) {
        fprintf(stderr, "[V1] malloc failed\n"); exit(EXIT_FAILURE);
    }

    CudaTimer timer;
    timer.start();

    double error = 1.0e30;
    int iter = 0;
    bool converged = false;

    while (iter < maxi && !converged) {
        ++iter;

        // Launch kernel.
        jacobi_kernel_naive<<<grid, block>>>(d_u_new, d_u, d_f, N, h2);

        // Swap device pointers: d_u_new (newly computed) becomes current.
        double* tmp = d_u_new;
        d_u_new = d_u;
        d_u     = tmp;

        // Strategy A: check convergence every check_every iterations.
        // Copy both grids to host; compute max|diff| on CPU.
        // Cost: 2 × N² doubles per check — expensive for large N,
        //       but simple and sufficient for a first GPU version.
        if (iter % check_every == 0) {
            CUDA_CHECK(cudaMemcpy(h_u,     d_u,     bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_u_new, d_u_new, bytes, cudaMemcpyDeviceToHost));

            // h_u = current (iter k),  h_u_new = previous (iter k-1 after swap).
            error = 0.0;
            for (int k = 0; k < N * N; ++k) {
                const double d = fabs(h_u[k] - h_u_new[k]);
                if (d > error) error = d;
            }
            converged = (error < tol);

            if (iter % (check_every * 10) == 0) {
                printf("[GPU V1]  iter = %6d   error = %.6e\n", iter, error);
            }
        }
    }

    timer.stop();
    free(h_u);
    free(h_u_new);

    SolverResult res;
    res.iters       = iter;
    res.final_error = error;
    res.total_ms    = timer.elapsed_ms();
    res.ms_per_iter = (iter > 0) ? (res.total_ms / iter) : 0.0;
    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// V2 Host-side solver wrapper
// ═════════════════════════════════════════════════════════════════════════════
SolverResult jacobi_gpu_optimized(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f)
{
    const int    N           = params.N;
    const double h2          = params.h * params.h;
    const int    maxi        = params.max_iter;
    const double tol         = params.tol;
    const int    check_every = params.check_every;

    dim3 block(TILE_X, TILE_Y);
    dim3 grid(
        (unsigned int)((N + TILE_X - 1) / TILE_X),
        (unsigned int)((N + TILE_Y - 1) / TILE_Y)
    );
    const int num_blocks = (int)(grid.x * grid.y);

    // Device array: one max-diff per block, written by the kernel every iter.
    double* d_block_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_max, (size_t)num_blocks * sizeof(double)));

    // Host mirror for reading block maxima.
    double* h_block_max = (double*)malloc((size_t)num_blocks * sizeof(double));
    if (!h_block_max) {
        fprintf(stderr, "[V2] malloc failed\n"); exit(EXIT_FAILURE);
    }

    CudaTimer timer;
    timer.start();

    double error = 1.0e30;
    int iter = 0;
    bool converged = false;

    while (iter < maxi && !converged) {
        ++iter;

        // Launch optimised kernel.
        // The kernel always writes per-block max to d_block_max.
        jacobi_kernel_shared<<<grid, block>>>(
            d_u_new, d_u, d_f, N, h2, d_block_max);

        // Swap device pointers.
        double* tmp = d_u_new;
        d_u_new = d_u;
        d_u     = tmp;

        // Strategy B: read per-block maxima every check_every iterations.
        // Transfer: num_blocks doubles (e.g. 1024 doubles = 8 KB for N=512).
        // Much cheaper than Strategy A (which would copy the entire grid).
        if (iter % check_every == 0) {
            CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max,
                                  (size_t)num_blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            error     = host_reduce_max(h_block_max, num_blocks);
            converged = (error < tol);

            if (iter % (check_every * 10) == 0) {
                printf("[GPU V2]  iter = %6d   error = %.6e\n", iter, error);
            }
        }
    }

    timer.stop();
    CUDA_CHECK(cudaFree(d_block_max));
    free(h_block_max);

    SolverResult res;
    res.iters       = iter;
    res.final_error = error;
    res.total_ms    = timer.elapsed_ms();
    res.ms_per_iter = (iter > 0) ? (res.total_ms / iter) : 0.0;
    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// V3 Host-side solver wrapper
// ═════════════════════════════════════════════════════════════════════════════
SolverResult jacobi_gpu_coalesced(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f)
{
    const int    N           = params.N;
    const double h2          = params.h * params.h;
    const int    maxi        = params.max_iter;
    const double tol         = params.tol;
    const int    check_every = params.check_every;

    dim3 block(TILE_X, TILE_Y);
    dim3 grid(
        (unsigned int)((N + TILE_X - 1) / TILE_X),
        (unsigned int)((N + TILE_Y - 1) / TILE_Y)
    );
    const int num_blocks = (int)(grid.x * grid.y);

    // Device array: one max-diff per block.
    double* d_block_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_max, (size_t)num_blocks * sizeof(double)));

    // Host mirror.
    double* h_block_max = (double*)malloc((size_t)num_blocks * sizeof(double));
    if (!h_block_max) {
        fprintf(stderr, "[V3] malloc failed\n"); exit(EXIT_FAILURE);
    }

    CudaTimer timer;
    timer.start();

    double error = 1.0e30;
    int iter = 0;
    bool converged = false;

    while (iter < maxi && !converged) {
        ++iter;

        // Launch coalesced kernel (V3).
        jacobi_kernel_shared_coalesced<<<grid, block>>>(
            d_u_new, d_u, d_f, N, h2, d_block_max);

        // Swap device pointers.
        double* tmp = d_u_new;
        d_u_new = d_u;
        d_u     = tmp;

        if (iter % check_every == 0) {
            CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max,
                                  (size_t)num_blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            error     = host_reduce_max(h_block_max, num_blocks);
            converged = (error < tol);

            if (iter % (check_every * 10) == 0) {
                printf("[GPU V3]  iter = %6d   error = %.6e\n", iter, error);
            }
        }
    }

    timer.stop();
    CUDA_CHECK(cudaFree(d_block_max));
    free(h_block_max);

    SolverResult res;
    res.iters       = iter;
    res.final_error = error;
    res.total_ms    = timer.elapsed_ms();
    res.ms_per_iter = (iter > 0) ? (res.total_ms / iter) : 0.0;
    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// Pure-throughput benchmark (no convergence check, no H↔D transfer)
// ═════════════════════════════════════════════════════════════════════════════
float jacobi_gpu_benchmark(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f,
    int           bench_iters,
    int           version)
{
    const int    N  = params.N;
    const double h2 = params.h * params.h;

    dim3 block(TILE_X, TILE_Y);
    dim3 grid(
        (unsigned int)((N + TILE_X - 1) / TILE_X),
        (unsigned int)((N + TILE_Y - 1) / TILE_Y)
    );
    const int num_blocks = (int)(grid.x * grid.y);

    // Dummy d_block_max needed by V2/V3 kernels (not read by this benchmark).
    double* d_block_max = nullptr;
    if (version > 0) {
        CUDA_CHECK(cudaMalloc(&d_block_max, (size_t)num_blocks * sizeof(double)));
    }

    // Warmup: one iteration to prime caches and avoid cold-start bias.
    if (version == 2)
        jacobi_kernel_shared_coalesced<<<grid, block>>>(d_u_new, d_u, d_f, N, h2, d_block_max);
    else if (version == 1)
        jacobi_kernel_shared<<<grid, block>>>(d_u_new, d_u, d_f, N, h2, d_block_max);
    else
        jacobi_kernel_naive <<<grid, block>>>(d_u_new, d_u, d_f, N, h2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed benchmark loop.
    CudaTimer timer;
    timer.start();

    for (int it = 0; it < bench_iters; ++it) {
        if (version == 2)
            jacobi_kernel_shared_coalesced<<<grid, block>>>(d_u_new, d_u, d_f, N, h2, d_block_max);
        else if (version == 1)
            jacobi_kernel_shared<<<grid, block>>>(d_u_new, d_u, d_f, N, h2, d_block_max);
        else
            jacobi_kernel_naive <<<grid, block>>>(d_u_new, d_u, d_f, N, h2);

        // Swap pointers to maintain double-buffer semantics.
        double* tmp = d_u_new;
        d_u_new = d_u;
        d_u     = tmp;
    }

    timer.stop();

    if (version > 0) CUDA_CHECK(cudaFree(d_block_max));

    return timer.elapsed_ms();
}


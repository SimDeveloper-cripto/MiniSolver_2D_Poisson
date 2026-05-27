#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ── Grid indexing
#define IDX(i, j, N)  ((i) * (N) + (j))

// ── Tile / block dimensions
// 16 × 16 = 256 threads/block
#define TILE_X  16
#define TILE_Y  16

// Shared-memory tile size
#define SMEM_X  (TILE_X + 2)
#define SMEM_Y  (TILE_Y + 2)

// How often convergence is checked
#define DEFAULT_CHECK_EVERY  100

// ── CUDA error-checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t _e = (call);                                  \
        if (_e != cudaSuccess) {                                  \
            fprintf(stderr,                                       \
                    "[CUDA ERROR] %s  line %d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));  \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// ── Solver Config
struct SolverParams {
    int    N;            // Grid size is N × N (includes Dirichlet boundary)
    int    max_iter;     // Max # Jacobi iterations
    int    check_every;  // Convergence-check frequency (iterations)

    double h;            // Grid spacing = 1 / (N - 1)
    double tol;          // Convergence tolerance on max|u_new - u_old|
};

// ── Solver Result
struct SolverResult {
    int    iters;        // # Jacobi iterations performed

    double final_error;
    double total_ms;     // includes H↔D xfer for convergence checks in V1; excludes alloc.
    double ms_per_iter;
};
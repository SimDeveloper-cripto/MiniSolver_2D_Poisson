#include <cmath>

#include <vector>
#include <cstdio>

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "../include/timer.h"
#include "../include/common.h"
#include "../include/cpu_solver.h"
#include "../include/validation.h"

#include "../include/reduction.h"
#include "../include/gpu_solver.h"
#include "../include/solver_runner.h"

void align_grid_size(int& N) {
    if (N % TILE_X != 0 || N % TILE_Y != 0) {
        printf("[WARNING] N (%d) is not a multiple of TILE size (%dx%d).\n", N, TILE_X, TILE_Y);
        printf("          Rounding N up to next multiple of %d...\n", TILE_X);

        N = ((N + TILE_X - 1) / TILE_X) * TILE_X;
        printf("          New N = %d\n\n", N);
    }
}

FullSolverResults run_full_solve(const SolverParams& params, bool run_cpu) {
    FullSolverResults results = {};
    results.run_cpu           = run_cpu;

    const int N        = params.N;
    const double h     = params.h;
    const size_t bytes = (size_t)N * N * sizeof(double);

    // Host memory allocation
    double* h_u_cpu    = (double*)malloc(bytes);
    double* h_u_new    = (double*)malloc(bytes);
    double* h_f        = (double*)malloc(bytes);
    double* h_u_gpu_v1 = (double*)malloc(bytes);
    double* h_u_gpu_v2 = (double*)malloc(bytes);
    double* h_u_gpu_v3 = (double*)malloc(bytes);

    if (!h_u_cpu || !h_u_new || !h_f || !h_u_gpu_v1 || !h_u_gpu_v2 || !h_u_gpu_v3) {
        fprintf(stderr, "Host malloc failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize source term and boundaries
    initialize(h_u_cpu, h_f, N, h);
    std::memcpy(h_u_new,    h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v1, h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v2, h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v3, h_u_cpu, bytes);

    // Device memory allocation
    double *d_u      = nullptr;
    double *d_u_new  = nullptr;
    double *d_f      = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u,     bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_f,     bytes));

    // ── 1. CPU Sequential Solve ──────────────────────────────────────────────
    if (run_cpu) {
        printf("Running CPU Jacobi Solver...\n");
        double* h_u_cpu_run = (double*)malloc(bytes);
        double* h_u_new_run = (double*)malloc(bytes);
        std::memcpy(h_u_cpu_run, h_u_cpu, bytes);
        std::memcpy(h_u_new_run, h_u_cpu, bytes);

        results.cpu_res = jacobi_cpu(params, h_u_cpu_run, h_u_new_run, h_f);

        std::memcpy(h_u_cpu, (results.cpu_res.iters % 2 == 1) ? h_u_new_run : h_u_cpu_run, bytes);

        free(h_u_cpu_run);
        free(h_u_new_run);

        printf("  CPU: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
               results.cpu_res.iters, results.cpu_res.total_ms, results.cpu_res.ms_per_iter, results.cpu_res.final_error);
    } else {
        printf("Skipping CPU Solver.\n\n");
    }

    // ── 2. GPU Naive (V1) Solve ──────────────────────────────────────────────
    printf("Running GPU Naive (V1) Solver (Strategy A: H<->D copying)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v1,     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, h_f,            bytes, cudaMemcpyHostToDevice));

    results.v1_res = jacobi_gpu_naive(params, d_u, d_u_new, d_f);

    CUDA_CHECK(cudaMemcpy(h_u_gpu_v1, (results.v1_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V1: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           results.v1_res.iters, results.v1_res.total_ms, results.v1_res.ms_per_iter, results.v1_res.final_error);

    // ── 3. GPU Optimized (V2) Solve ──────────────────────────────────────────
    printf("Running GPU Optimized (V2) Solver (Strategy B: Shared Memory + Block Reduction)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v2,     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v2, bytes, cudaMemcpyHostToDevice));

    results.v2_res = jacobi_gpu_optimized(params, d_u, d_u_new, d_f);

    CUDA_CHECK(cudaMemcpy(h_u_gpu_v2, (results.v2_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V2: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           results.v2_res.iters, results.v2_res.total_ms, results.v2_res.ms_per_iter, results.v2_res.final_error);

    // ── 3.1. GPU Coalesced (V3) Solve ─────────────────────────────────────────
    printf("Running GPU Coalesced (V3) Solver (Strategy B: Coalesced Loader + Block Reduction)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v3, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v3, bytes, cudaMemcpyHostToDevice));

    results.v3_res = jacobi_gpu_coalesced(params, d_u, d_u_new, d_f);

    CUDA_CHECK(cudaMemcpy(h_u_gpu_v3, (results.v3_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V3: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           results.v3_res.iters, results.v3_res.total_ms, results.v3_res.ms_per_iter, results.v3_res.final_error);

    // Calculate validation metrics
    const int n = N * N;
    if (run_cpu) {
        results.v1_vs_cpu_max_diff = max_abs_diff(h_u_cpu, h_u_gpu_v1, n);
        results.v1_vs_cpu_rms      = rms_diff(h_u_cpu, h_u_gpu_v1, n);

        results.v2_vs_cpu_max_diff = max_abs_diff(h_u_cpu, h_u_gpu_v2, n);
        results.v2_vs_cpu_rms      = rms_diff(h_u_cpu, h_u_gpu_v2, n);

        results.v3_vs_cpu_max_diff = max_abs_diff(h_u_cpu, h_u_gpu_v3, n);
        results.v3_vs_cpu_rms      = rms_diff(h_u_cpu, h_u_gpu_v3, n);

        results.cpu_analytical_err = max_error_vs_exact(h_u_cpu, N, h);
    }

    results.v1_analytical_err = max_error_vs_exact(h_u_gpu_v1, N, h);
    results.v2_analytical_err = max_error_vs_exact(h_u_gpu_v2, N, h);
    results.v3_analytical_err = max_error_vs_exact(h_u_gpu_v3, N, h);

    // Clean-Up memory
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_new));
    CUDA_CHECK(cudaFree(d_f));

    free(h_u_cpu);
    free(h_u_new);
    free(h_f);
    free(h_u_gpu_v1);
    free(h_u_gpu_v2);
    free(h_u_gpu_v3);

    return results;
}

void print_solver_results(const FullSolverResults& results, const SolverParams& params) {
    const double tol = params.tol;
    printf("=========================================================\n");
    printf("  VALIDATION AND CORRECTNESS REPORT\n");
    printf("=========================================================\n");

    if (results.run_cpu) {
        const bool v1_pass = (results.v1_vs_cpu_max_diff < tol * 100.0);
        printf("  %-18s  max|diff|=%.3e  rms=%.3e  %s\n",
               "GPU V1 vs CPU", results.v1_vs_cpu_max_diff, results.v1_vs_cpu_rms, v1_pass ? "[ PASS ]" : "[ FAIL ]");

        const bool v2_pass = (results.v2_vs_cpu_max_diff < tol * 100.0);
        printf("  %-18s  max|diff|=%.3e  rms=%.3e  %s\n",
               "GPU V2 vs CPU", results.v2_vs_cpu_max_diff, results.v2_vs_cpu_rms, v2_pass ? "[ PASS ]" : "[ FAIL ]");

        const bool v3_pass = (results.v3_vs_cpu_max_diff < tol * 100.0);
        printf("  %-18s  max|diff|=%.3e  rms=%.3e  %s\n",
               "GPU V3 vs CPU", results.v3_vs_cpu_max_diff, results.v3_vs_cpu_rms, v3_pass ? "[ PASS ]" : "[ FAIL ]");

        printf("  %-18s  max|error|=%.3e\n", "CPU vs Exact", results.cpu_analytical_err);
    }

    printf("  %-18s  max|error|=%.3e\n", "GPU V1 vs Exact", results.v1_analytical_err);
    printf("  %-18s  max|error|=%.3e\n", "GPU V2 vs Exact", results.v2_analytical_err);
    printf("  %-18s  max|error|=%.3e\n", "GPU V3 vs Exact", results.v3_analytical_err);
    printf("=========================================================\n\n");
}

bool test_standalone_reduction() {
    printf("--- Running Standalone GPU Max-Reduction Test ---\n");
    const int M = 2500000; // 2.5 Million elements
    std::vector<double> h_test(M);

    // Initialize with random numbers in [0, 100]
    double cpu_max = -1.0;
    for (int i = 0; i < M; ++i) {
        h_test[i] = (double)rand() / RAND_MAX * 100.0;
        if (h_test[i] > cpu_max) cpu_max = h_test[i];
    }

    // Force a known maximum at a random index
    const int special_idx    = rand() % M;
    const double special_val = 9999.87654;
    h_test[special_idx]      = special_val;
    if (special_val > cpu_max) cpu_max = special_val;

    // Allocate and Copy to GPU
    double* d_test = nullptr;
    CUDA_CHECK(cudaMalloc(&d_test, M * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_test, h_test.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    // Run GPU Reduction
    CpuTimer timer;
    timer.start();
    double gpu_max = reduce_max_gpu(d_test, M);
    timer.stop();

    CUDA_CHECK(cudaFree(d_test));

    const double diff = std::fabs(cpu_max - gpu_max);
    const bool pass   = (diff < 1e-9);

    printf("  Vector Size : %d elements (%.2f MB)\n", M, (double)(M * sizeof(double)) / (1024.0 * 1024.0));
    printf("  CPU Max     : %.5f\n", cpu_max);
    printf("  GPU Max     : %.5f\n", gpu_max);
    printf("  Difference  : %.3e\n", diff);
    printf("  Time (GPU)  : %.3f ms\n", timer.elapsed_ms());
    printf("  Result      : %s\n\n", pass ? "[ PASS ]" : "[ FAIL ]");

    return pass;
}
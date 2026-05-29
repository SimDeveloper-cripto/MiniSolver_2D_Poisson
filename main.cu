// Compile:
// !nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu src/validation.cpp main.cu -o minisolver

// Execute (need to set parameters):
// !./minisolver --n 256 --tol 1e-7

// TODO:
// Refactor main.cu []

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "include/timer.h"
#include "include/common.h"

#include "include/cpu_solver.h"
#include "include/validation.h"

#include "include/reduction.h"
#include "include/gpu_solver.h"

void print_gpu_properties() {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        printf("No CUDA-capable GPU detected or CUDA runtime error.\n");
        return;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("=========================================================\n");
    printf("  GPU DEVICE PROPERTIES\n");
    printf("=========================================================\n");
    printf("  Device Name          : %s\n", prop.name);
    printf("  Compute Capability   : %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory        : %.2f GB\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Mem / Block   : %.2f KB\n", (double)prop.sharedMemPerBlock / 1024.0);
    printf("  Warp Size            : %d\n", prop.warpSize);
    printf("  Max Threads / Block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Max Grid Dimensions  : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max Block Dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("=========================================================\n\n");
}

// ── Test & Validate the Standalone Parallel Reduction
void test_standalone_reduction() {
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
}

int main(int argc, char** argv) {
    // Default Solver Params
    int N                = 256;
    double tol           = 1.0e-7;
    int max_iter         = 50000;
    int check_every      = DEFAULT_CHECK_EVERY;
    bool run_scalability = true;
    bool run_cpu         = true;

    // Command-line parsing
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--n" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--tol" && i + 1 < argc) {
            tol = std::atof(argv[++i]);
        }
        else if (std::string(argv[i]) == "--max-iter" && i + 1 < argc) {
            max_iter = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--check-every" && i + 1 < argc) {
            check_every = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--no-scalability") {
            run_scalability = false;
        }
        else if (std::string(argv[i]) == "--no-cpu") {
            run_cpu = false;
        }
        else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --n <int>           Grid size N (default: 256)\n");
            printf("  --tol <double>      Convergence Tolerance (default: 1.0e-7)\n");
            printf("  --max-iter <int>    Max Jacobi Iterations (default: 50000)\n");
            printf("  --check-every <int> Iterations between checks (default: 100)\n");
            printf("  --no-scalability    Skip the Scalability Benchmark suite\n");
            printf("  --no-cpu            Skip the sequential CPU execution\n");
            return 0;
        }
    }

    // N must be a multiple of TILE size for boundary safety in V2
    if (N % TILE_X != 0 || N % TILE_Y != 0) {
        printf("[WARNING] N (%d) is not a multiple of TILE size (%dx%d).\n", N, TILE_X, TILE_Y);
        printf("          Rounding N up to next multiple of %d...\n", TILE_X);

        // Trick to round up to the nearest multiple of TILE_X
        N = ((N + TILE_X - 1) / TILE_X) * TILE_X;
        printf("          New N = %d\n\n", N);
    }

    double h = 1.0 / (N - 1);
    SolverParams params = { N, h, tol, max_iter, check_every };

    print_gpu_properties();

    test_standalone_reduction();

    printf("=========================================================\n");
    printf("  FULL JACOBI POISSON SOLVE (N = %d, tol = %.1e)\n", N, tol);
    printf("=========================================================\n");

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
        return EXIT_FAILURE;
    }

    // Initialize source term and boundaries
    initialize(h_u_cpu, h_f, N, h);
    std::memcpy(h_u_new,    h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v1, h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v2, h_u_cpu, bytes);
    std::memcpy(h_u_gpu_v3, h_u_cpu, bytes);

    // Device memory allocation
    double *d_u     = nullptr;
    double *d_u_new = nullptr;
    double*d_f      = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_f, bytes));

    // ── 1. CPU Sequential Solve ──────────────────────────────────────────────
    SolverResult cpu_res = {0, 0.0, 0.0, 0.0};
    if (run_cpu) {
        printf("Running CPU Jacobi Solver...\n");
        // Create copies of the initial state because solver swaps pointers internally
        double* h_u_cpu_run = (double*)malloc(bytes);
        double* h_u_new_run = (double*)malloc(bytes);
        std::memcpy(h_u_cpu_run, h_u_cpu, bytes);
        std::memcpy(h_u_new_run, h_u_cpu, bytes); // starts from 0

        cpu_res = jacobi_cpu(params, h_u_cpu_run, h_u_new_run, h_f);

        // Copy back the correct final buffer based on odd/even iterations
        std::memcpy(h_u_cpu, (cpu_res.iters % 2 == 1) ? h_u_new_run : h_u_cpu_run, bytes);

        free(h_u_cpu_run);
        free(h_u_new_run);

        printf("  CPU: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
               cpu_res.iters, cpu_res.total_ms, cpu_res.ms_per_iter, cpu_res.final_error);
    } else {
        printf("Skipping CPU Solver.\n\n");
    }

    // ── 2. GPU Naive (V1) Solve ──────────────────────────────────────────────
    printf("Running GPU Naive (V1) Solver (Strategy A: H<->D copying)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v1,     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, h_f,            bytes, cudaMemcpyHostToDevice));

    SolverResult v1_res = jacobi_gpu_naive(params, d_u, d_u_new, d_f);

    // Copy back final solution (respecting swap)
    CUDA_CHECK(cudaMemcpy(h_u_gpu_v1, (v1_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V1: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           v1_res.iters, v1_res.total_ms, v1_res.ms_per_iter, v1_res.final_error);

    // ── 3. GPU Optimized (V2) Solve ──────────────────────────────────────────
    printf("Running GPU Optimized (V2) Solver (Strategy B: Shared Memory + Block Reduction)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v2,     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v2, bytes, cudaMemcpyHostToDevice));

    SolverResult v2_res = jacobi_gpu_optimized(params, d_u, d_u_new, d_f);

    // Copy back final solution (respecting swap)
    CUDA_CHECK(cudaMemcpy(h_u_gpu_v2, (v2_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V2: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           v2_res.iters, v2_res.total_ms, v2_res.ms_per_iter, v2_res.final_error);

    // ── 3.1. GPU Coalesced (V3) Solve ─────────────────────────────────────────
    printf("Running GPU Coalesced (V3) Solver (Strategy B: Coalesced Loader + Block Reduction)...\n");
    CUDA_CHECK(cudaMemcpy(d_u, h_u_gpu_v3, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u_gpu_v3, bytes, cudaMemcpyHostToDevice));

    SolverResult v3_res = jacobi_gpu_coalesced(params, d_u, d_u_new, d_f);

    // Copy back final solution (respecting swap)
    CUDA_CHECK(cudaMemcpy(h_u_gpu_v3, (v3_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));

    printf("  GPU V3: %d iterations in %.2f ms (%.4f ms/iter). Final err: %.6e\n\n",
           v3_res.iters, v3_res.total_ms, v3_res.ms_per_iter, v3_res.final_error);


    // ── 4. Validation ────────────────────────────────────────────────────────
    printf("=========================================================\n");
    printf("  VALIDATION AND CORRECTNESS REPORT\n");
    printf("=========================================================\n");
    if (run_cpu) {
        print_validation(h_u_cpu, h_u_gpu_v1, N, tol, "GPU V1 vs CPU");
        print_validation(h_u_cpu, h_u_gpu_v2, N, tol, "GPU V2 vs CPU");
        print_validation(h_u_cpu, h_u_gpu_v3, N, tol, "GPU V3 vs CPU");

        double cpu_analytical_err = max_error_vs_exact(h_u_cpu, N, h);
        printf("  %-18s  max|error|=%.3e\n", "CPU vs Exact", cpu_analytical_err);
    }
    double v1_analytical_err = max_error_vs_exact(h_u_gpu_v1, N, h);
    double v2_analytical_err = max_error_vs_exact(h_u_gpu_v2, N, h);
    double v3_analytical_err = max_error_vs_exact(h_u_gpu_v3, N, h);
    printf("  %-18s  max|error|=%.3e\n", "GPU V1 vs Exact", v1_analytical_err);
    printf("  %-18s  max|error|=%.3e\n", "GPU V2 vs Exact", v2_analytical_err);
    printf("  %-18s  max|error|=%.3e\n", "GPU V3 vs Exact", v3_analytical_err);
    printf("=========================================================\n\n");

    // Clean up temporary host/device memory used in full solves
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_new));
    CUDA_CHECK(cudaFree(d_f));

    free(h_u_cpu);
    free(h_u_new);
    free(h_u_gpu_v1);
    free(h_u_gpu_v2);
    free(h_u_gpu_v3);

    // ── 5. Scalability Benchmark Suite ────────────────────────────────────────
    if (run_scalability) {
        printf("=========================================================\n");
        printf("  SCALABILITY BENCHMARK SUITE (1000 Iterations)\n");
        printf("=========================================================\n");
        printf("%-5s | %-12s | %-12s | %-12s | %-10s | %-8s\n",
               "N", "Solver", "Time (ms)", "MUpdates/s", "GB/s (Ideal)", "Speedup");
        printf("---------------------------------------------------------\n");

        std::vector<int> sizes = {128, 256, 512, 1024};
        const int bench_iters = 1000;

        for (int bn : sizes) {
            double bh = 1.0 / (bn - 1);
            size_t bbytes = (size_t)bn * bn * sizeof(double);

            // Alloc
            double* hb_u = (double*)malloc(bbytes);
            double* hb_f = (double*)malloc(bbytes);
            initialize(hb_u, hb_f, bn, bh);

            double *db_u = nullptr, *db_u_new = nullptr, *db_f = nullptr;
            CUDA_CHECK(cudaMalloc(&db_u, bbytes));
            CUDA_CHECK(cudaMalloc(&db_u_new, bbytes));
            CUDA_CHECK(cudaMalloc(&db_f, bbytes));

            CUDA_CHECK(cudaMemcpy(db_u, hb_u, bbytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db_u_new, hb_u, bbytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db_f, hb_f, bbytes, cudaMemcpyHostToDevice));

            // CPU Benchmark (skip or do fewer iterations for N = 1024 to save time, say 100 and scale)
            double cpu_t_ms = 0.0;
            if (run_cpu) {
                int cpu_bench_iters = (bn <= 256) ? bench_iters : 100;
                double* hb_u_new = (double*)malloc(bbytes);
                std::memcpy(hb_u_new, hb_u, bbytes);

                CpuTimer cpu_timer;
                cpu_timer.start();

                double* u_ptr  = hb_u;
                double* un_ptr = hb_u_new;
                double bh2     = bh * bh;
                for (int it = 0; it < cpu_bench_iters; ++it) {
                    jacobi_step_cpu(un_ptr, u_ptr, hb_f, bn, bh2);

                    double* tmp = un_ptr;
                    un_ptr      = u_ptr;
                    u_ptr       = tmp;
                }
                cpu_timer.stop();
                cpu_t_ms = cpu_timer.elapsed_ms() * ((double)bench_iters / cpu_bench_iters);

                free(hb_u_new);

                double cpu_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (cpu_t_ms * 1000.0);
                double cpu_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (cpu_t_ms * 1.0e6);
                printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %-8s\n",
                       bn, "CPU", cpu_t_ms, cpu_mupdates, cpu_gb, "1.00x (Ref)");
            }

            // GPU Naive (V1)
            float v1_t_ms      = jacobi_gpu_benchmark({bn, bh, 0.0, 0, 0}, db_u, db_u_new, db_f, bench_iters, 0);
            double v1_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (v1_t_ms * 1000.0);
            double v1_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (v1_t_ms * 1.0e6);
            double v1_speedup  = (run_cpu) ? (cpu_t_ms / v1_t_ms) : 1.0;

            printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %7.2fx\n",
                   bn, "GPU V1 Naive", (double)v1_t_ms, v1_mupdates, v1_gb, v1_speedup);

            // GPU Optimized (V2)
            float v2_t_ms      = jacobi_gpu_benchmark({bn, bh, 0.0, 0, 0}, db_u, db_u_new, db_f, bench_iters, 1);
            double v2_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (v2_t_ms * 1000.0);
            double v2_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (v2_t_ms * 1.0e6);
            double v2_speedup  = (run_cpu) ? (cpu_t_ms / v2_t_ms) : 1.0;
            double v2_vs_v1    = (double)v1_t_ms / v2_t_ms;

            printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %7.2fx (%5.2fx vs V1)\n",
                   bn, "GPU V2 Opt", (double)v2_t_ms, v2_mupdates, v2_gb, v2_speedup, v2_vs_v1);

            // GPU Coalesced (V3)
            float v3_t_ms      = jacobi_gpu_benchmark({bn, bh, 0.0, 0, 0}, db_u, db_u_new, db_f, bench_iters, 2);
            double v3_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (v3_t_ms * 1000.0);
            double v3_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (v3_t_ms * 1.0e6);
            double v3_speedup  = (run_cpu) ? (cpu_t_ms / v3_t_ms) : 1.0;
            double v3_vs_v2    = (double)v2_t_ms / v3_t_ms;

            printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %7.2fx (%5.2fx vs V2)\n",
                   bn, "GPU V3 Coal", (double)v3_t_ms, v3_mupdates, v3_gb, v3_speedup, v3_vs_v2);
            printf("---------------------------------------------------------\n");

            // Free
            CUDA_CHECK(cudaFree(db_u));
            CUDA_CHECK(cudaFree(db_u_new));
            CUDA_CHECK(cudaFree(db_f));
            free(hb_u);
            free(hb_f);
        }
        printf("=========================================================\n\n");
    }

    free(h_f);
    return 0;
}
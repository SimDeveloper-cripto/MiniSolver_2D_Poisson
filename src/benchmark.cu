#include <vector>
#include <cstdio>

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "../include/timer.h"
#include "../include/common.h"
#include "../include/benchmark.h"

#include "../include/cpu_solver.h"
#include "../include/gpu_solver.h"

void run_scalability_benchmark(const std::vector<int>& sizes, bool run_cpu) {
    printf("=========================================================\n");
    printf("  SCALABILITY BENCHMARK SUITE (1000 Iterations)\n");
    printf("=========================================================\n");
    printf("%-5s | %-12s | %-12s | %-12s | %-10s | %-8s\n",
           "N", "Solver", "Time (ms)", "MUpdates/s", "GB/s (Ideal)", "Speedup");
    printf("---------------------------------------------------------\n");

    const int bench_iters = 1000;

    for (int bn : sizes) {
        double bh     = 1.0 / (bn - 1);
        size_t bbytes = (size_t)bn * bn * sizeof(double);

        // Alloc
        double* hb_u = (double*)malloc(bbytes);
        double* hb_f = (double*)malloc(bbytes);
        if (!hb_u || !hb_f) {
            fprintf(stderr, "Host allocation failed in benchmark.\n");
            exit(EXIT_FAILURE);
        }
        initialize(hb_u, hb_f, bn, bh);

        double *db_u     = nullptr;
        double *db_u_new = nullptr;
        double *db_f     = nullptr;
        CUDA_CHECK(cudaMalloc(&db_u,     bbytes));
        CUDA_CHECK(cudaMalloc(&db_u_new, bbytes));
        CUDA_CHECK(cudaMalloc(&db_f,     bbytes));

        CUDA_CHECK(cudaMemcpy(db_u,     hb_u, bbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db_u_new, hb_u, bbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db_f,     hb_f, bbytes, cudaMemcpyHostToDevice));

        // CPU Benchmark
        // (skip or do fewer iterations for N = 1024 to save time)
        double cpu_t_ms = 0.0;
        if (run_cpu) {
            int cpu_bench_iters = (bn <= 256) ? bench_iters : 100;
            double* hb_u_new = (double*)malloc(bbytes);
            if (!hb_u_new) {
                fprintf(stderr, "Host allocation failed in CPU benchmark.\n");
                exit(EXIT_FAILURE);
            }
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

        // GPU Benchmarks configuration
        SolverParams bench_params;
        bench_params.N           = bn;
        bench_params.max_iter    = 0;
        bench_params.check_every = 0;
        bench_params.h           = bh;
        bench_params.tol         = 0.0;

        // GPU Naive (V1)
        float v1_t_ms      = jacobi_gpu_benchmark(bench_params, db_u, db_u_new, db_f, bench_iters, 0);
        double v1_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (v1_t_ms * 1000.0);
        double v1_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (v1_t_ms * 1.0e6);
        double v1_speedup  = (run_cpu) ? (cpu_t_ms / v1_t_ms) : 1.0;

        printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %7.2fx\n",
               bn, "GPU V1 Naive", (double)v1_t_ms, v1_mupdates, v1_gb, v1_speedup);

        // GPU Optimized (V2)
        float v2_t_ms      = jacobi_gpu_benchmark(bench_params, db_u, db_u_new, db_f, bench_iters, 1);
        double v2_mupdates = (double)(bn - 2) * (bn - 2) * bench_iters / (v2_t_ms * 1000.0);
        double v2_gb       = (3.0 * sizeof(double) * (bn - 2) * (bn - 2) * bench_iters) / (v2_t_ms * 1.0e6);
        double v2_speedup  = (run_cpu) ? (cpu_t_ms / v2_t_ms) : 1.0;
        double v2_vs_v1    = (double)v1_t_ms / v2_t_ms;

        printf("%-5d | %-12s | %12.2f | %12.2f | %12.2f | %7.2fx (%5.2fx vs V1)\n",
               bn, "GPU V2 Opt", (double)v2_t_ms, v2_mupdates, v2_gb, v2_speedup, v2_vs_v1);

        // GPU Coalesced (V3)
        float v3_t_ms      = jacobi_gpu_benchmark(bench_params, db_u, db_u_new, db_f, bench_iters, 2);
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
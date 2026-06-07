// src/benchmark.cu

// [1] Williams et al. (2009). "Roofline". CACM 52(4), 65-76.
// [2] NVIDIA T4 Datasheet. docs.nvidia.com
// [3] Volkov (2010). "Better Performance at Lower Occupancy". GTC.
// [4] Kirk & Hwu (2016). "Programming Massively Parallel Processors." Elsevier.

#include <cmath>
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


/**
 * gpu_bench_avg – run n_runs independent calls to jacobi_gpu_benchmark,
 * return mean ± σ over the run times.
 */
static BenchStats gpu_bench_avg(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f,
    int           n_runs,
    int           bench_iters,
    int           version          // 0 = V1, 1 = V2, 2 = V3
) {
    double sum = 0.0, sum2 = 0.0;
    for (int r = 0; r < n_runs; ++r) {
        // jacobi_gpu_benchmark takes d_u / d_u_new by value (local copies),
        // so swapping inside the function does NOT affect the caller's arrays.
        float t = jacobi_gpu_benchmark(params, d_u, d_u_new, d_f, bench_iters, version);
        sum  += (double)t;
        sum2 += (double)t * (double)t;
    }
    double mean = sum / n_runs;
    double var  = (n_runs > 1) ? (sum2 / n_runs - mean * mean) : 0.0;
    BenchStats s;
    s.mean_ms = mean;
    s.std_ms  = (var > 0.0) ? sqrt(var) : 0.0;
    return s;
}

/**
 * gpu_bench_avg_bs – same as above but uses jacobi_gpu_benchmark_blocksize
 * with custom block dimensions bx × by.
 */
static BenchStats gpu_bench_avg_bs(
    SolverParams  params,
    double*       d_u,
    double*       d_u_new,
    const double* d_f,
    int           n_runs,
    int           bench_iters,
    int           bx,
    int           by
) {
    double sum = 0.0, sum2 = 0.0;
    for (int r = 0; r < n_runs; ++r) {
        float t = jacobi_gpu_benchmark_blocksize(params, d_u, d_u_new, d_f,
                                                  bench_iters, bx, by);
        sum  += (double)t;
        sum2 += (double)t * (double)t;
    }
    double mean = sum / n_runs;
    double var  = (n_runs > 1) ? (sum2 / n_runs - mean * mean) : 0.0;
    BenchStats s;
    s.mean_ms = mean;
    s.std_ms  = (var > 0.0) ? sqrt(var) : 0.0;
    return s;
}

/**
 * cpu_bench_avg – run n_runs independent loops of bench_iters calls to
 * jacobi_step_cpu; return mean ± σ over the run times.
 *
 * Using jacobi_step_cpu (single step) rather than the full jacobi_cpu
 * (run-to-convergence) gives a fair, fixed-iteration throughput comparison
 * with the GPU benchmark.
 */
static BenchStats cpu_bench_avg(
    const SolverParams& params,
    int                 n_runs,
    int                 bench_iters
) {
    const int    N     = params.N;
    const double h2    = params.h * params.h;
    const size_t bytes = (size_t)N * N * sizeof(double);

    double* h_u     = (double*)malloc(bytes);
    double* h_u_new = (double*)malloc(bytes);
    double* h_f     = (double*)malloc(bytes);
    if (!h_u || !h_u_new || !h_f) {
        fprintf(stderr, "[cpu_bench_avg] malloc failed\n"); exit(EXIT_FAILURE);
    }
    initialize(h_u, h_f, N, params.h);
    memcpy(h_u_new, h_u, bytes);

    double sum = 0.0, sum2 = 0.0;
    for (int r = 0; r < n_runs; ++r) {
        CpuTimer timer;
        timer.start();
        for (int it = 0; it < bench_iters; ++it) {
            jacobi_step_cpu(h_u_new, h_u, h_f, N, h2);
            double* tmp = h_u_new; h_u_new = h_u; h_u = tmp;
        }
        timer.stop();
        double t = (double)timer.elapsed_ms();
        sum  += t;
        sum2 += t * t;
    }

    free(h_u); free(h_u_new); free(h_f);

    double mean = sum / n_runs;
    double var  = (n_runs > 1) ? (sum2 / n_runs - mean * mean) : 0.0;
    BenchStats s;
    s.mean_ms = mean;
    s.std_ms  = (var > 0.0) ? sqrt(var) : 0.0;
    return s;
}

/**
 * gb_per_sec – effective bandwidth using the streaming lower bound.
 *   BW = 3 × N² × sizeof(double) × iters / time_s  [GB/s]
 *
 * The factor 3 accounts for 2 reads (u, f assumed L2-cached or 2 grid reads)
 * + 1 write (u_new), following the standard stencil BW model.
 * Ref: Williams et al. (2009), CACM 52(4), eq. (1).
 */
static double gb_per_sec(int N, double mean_ms, int iters) {
    double bytes = 3.0 * (double)N * (double)N * (double)sizeof(double) * iters;
    return bytes / (mean_ms * 1.0e6);   // bytes / (ms * 1e6) = GB/s
}

/**
 * gflops_per_sec – effective compute throughput.
 *   GFLOP/s = 6 × (N-2)² × iters / time_s
 *
 * 6 FLOPs per interior point: 4 additions + 1 multiply (h²·f) + 1 multiply (×0.25).
 */
static double gflops_per_sec(int N, double mean_ms, int iters) {
    double flops = 6.0 * (double)(N - 2) * (double)(N - 2) * iters;
    return flops / (mean_ms * 1.0e6);   // GFLOP/s
}


void print_roofline_analysis() {
    // ── NVIDIA T4 hardware specs [2] ─────────────────────────────────────────
    const double T4_FP64_TFLOPS = 0.260;   // FP64 Tensor Core boost, sm_75
    const double T4_BW_GBS      = 320.0;   // GDDR6 peak memory bandwidth [GB/s]
    const double ridge           = (T4_FP64_TFLOPS * 1000.0) / T4_BW_GBS; // FLOP/byte

    // ── 5-point Jacobi stencil analysis ─────────────────────────────────────
    // FLOPs/point: 4 additions (stencil sum) + 1 multiply (h²·f) + 1 multiply (×0.25)
    const double flops_pt = 6.0;

    // V1 streaming lower bound: 2 arrays read (u neighbours + f) + 1 write
    // = 3 × sizeof(double) = 24 bytes/point [Williams 2009, streaming bound]
    const double bytes_V1  = 3.0 * 8.0;   // = 24 bytes/point

    // V2/V3 with shared memory: the SMEM tile (TILE_X+2)×(TILE_Y+2) = 18×18 = 324
    // elements is loaded from global memory once per block (TILE_X×TILE_Y=256 points).
    // Effective reads from DRAM per element: 324/256 = 1.266 reads of u.
    // Plus f (1 read/element) and u_new write (1 write/element).
    // bytes_V2 = (18*18/256 + 1 + 1) * 8
    const double bytes_V2  = (18.0 * 18.0 / 256.0 + 2.0) * 8.0;  // ≈ 26.1 bytes/pt

    const double ai_V1 = flops_pt / bytes_V1;
    const double ai_V2 = flops_pt / bytes_V2;

    printf("=========================================================\n");
    printf("  ROOFLINE ANALYSIS  (NVIDIA Tesla T4, sm_75)\n");
    printf("=========================================================\n");
    printf("\n");
    printf("  Hardware\n");
    printf("    FP64 peak compute    : %.3f TFLOPS\n", T4_FP64_TFLOPS);
    printf("    Memory bandwidth     : %.0f GB/s (GDDR6)\n", T4_BW_GBS);
    printf("    Ridge point          : %.4f FLOP/byte\n", ridge);
    printf("\n");
    printf("  5-point Jacobi stencil\n");
    printf("    FLOPs / interior pt  : %.0f  (4 add, 1 mul h^2*f, 1 mul *0.25)\n",
           flops_pt);
    printf("\n");
    printf("  %-10s  %-28s  %12s  %-15s\n",
           "Kernel", "Bytes/pt  (method)", "AI (FLOP/B)", "Classification");
    printf("  ---------------------------------------------------------------\n");
    printf("  %-10s  %-28s  %12.4f  %s\n",
           "V1 naive",
           "3×8 B  (streaming lb)",
           ai_V1,
           (ai_V1 < ridge) ? "MEMORY-BOUND ◄" : "COMPUTE-BOUND");
    printf("  %-10s  %-28s  %12.4f  %s\n",
           "V2/V3 shmem",
           "(18^2/256+2)×8 B  (shmem)",
           ai_V2,
           (ai_V2 < ridge) ? "MEMORY-BOUND ◄" : "COMPUTE-BOUND");
    printf("\n");
    printf("  Ridge = %.4f FLOP/B   AI_V1 = %.4f FLOP/B   AI_V2 = %.4f FLOP/B\n",
           ridge, ai_V1, ai_V2);
    printf("  Both AI << ridge  ->  ALL variants are MEMORY-BOUND.\n");
    printf("\n");
    printf("  Peak achievable throughput (BW-limited):\n");
    printf("    V1  : %.1f GFLOP/s  (= %.0f GB/s * %.4f FLOP/B)\n",
           T4_BW_GBS * ai_V1, T4_BW_GBS, ai_V1);
    printf("    V2/3: %.1f GFLOP/s  (= %.0f GB/s * %.4f FLOP/B)\n",
           T4_BW_GBS * ai_V2, T4_BW_GBS, ai_V2);
    printf("\n");
    printf("  Ware-Amdahl expected GPU speedup vs CPU:\n");
    printf("    Serial fraction s -> 0 (pure stencil kernel).\n");
    printf("    CPU DRAM BW ~30-50 GB/s  ->  expected S = BW_GPU/BW_CPU = %.1f-%.1fx\n",
           T4_BW_GBS / 50.0, T4_BW_GBS / 30.0);
    printf("\n");
    printf("  References\n");
    printf("    [1] Williams, Waterman, Patterson (2009). \"Roofline: An Insightful\n");
    printf("        Visual Performance Model for Multicore Architectures.\"\n");
    printf("        Communications of the ACM, 52(4), pp. 65-76.\n");
    printf("    [2] NVIDIA (2023). Tesla T4 GPU Datasheet.\n");
    printf("        https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/\n");
    printf("        tesla-t4/t4-tensor-core-datasheet-951643.pdf\n");
    printf("=========================================================\n\n");
}


void run_scalability_benchmark(
    const std::vector<int>& sizes,
    bool                    run_cpu,
    bool                    csv_output
) {
    printf("=========================================================\n");
    printf("  SCALABILITY BENCHMARK\n");
    printf("  Fixed blocks: %dx%d | Runs: %d | Iters/run: %d\n",
           TILE_X, TILE_Y, N_BENCH_RUNS, BENCH_ITERS);
    printf("  BW = 3*N^2*8 B / t  [streaming lb, Williams et al. 2009]\n");
    printf("=========================================================\n");
    printf("%-6s  %-10s  %10s  %7s  %8s  %9s  %9s\n",
           "N", "Solver", "Mean(ms)", "Std(ms)", "GB/s", "GFLOP/s", "Speedup");
    printf("---------------------------------------------------------\n");

    if (csv_output) {
        printf("# BENCH_CSV format: tag,N,solver,bx,by,num_blocks,"
               "mean_ms,std_ms,gb_s,gflops,speedup_cpu,speedup_v1\n");
    }

    for (int N : sizes) {
        const double h     = 1.0 / (N - 1);
        const size_t bytes = (size_t)N * N * sizeof(double);
        const int num_blk  = ((N + TILE_X - 1) / TILE_X) *
                             ((N + TILE_Y - 1) / TILE_Y);

        SolverParams params;
        params.N           = N;
        params.h           = h;
        params.max_iter    = BENCH_ITERS;
        params.check_every = BENCH_ITERS + 1;   // never trigger convergence check
        params.tol         = 0.0;

        // ── Host alloc & init ─────────────────────────────────────────────────
        double* h_u = (double*)malloc(bytes);
        double* h_f = (double*)malloc(bytes);
        if (!h_u || !h_f) {
            fprintf(stderr, "[scalability] malloc failed\n"); exit(EXIT_FAILURE);
        }
        initialize(h_u, h_f, N, h);

        // ── Device alloc & copy ───────────────────────────────────────────────
        double *d_u = nullptr, *d_u_new = nullptr, *d_f = nullptr;
        CUDA_CHECK(cudaMalloc(&d_u,     bytes));
        CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
        CUDA_CHECK(cudaMalloc(&d_f,     bytes));
        CUDA_CHECK(cudaMemcpy(d_u,     h_u, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_new, h_u, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,     h_f, bytes, cudaMemcpyHostToDevice));

        // ── CPU baseline ──────────────────────────────────────────────────────
        BenchStats cpu = { 0.0, 0.0 };
        if (run_cpu) {
            int cpu_runs = (N > 512) ? N_CPU_RUNS_LARGE : N_BENCH_RUNS;
            cpu = cpu_bench_avg(params, cpu_runs, BENCH_ITERS);
            double cpu_gb = gb_per_sec(N, cpu.mean_ms, BENCH_ITERS);
            double cpu_gf = gflops_per_sec(N, cpu.mean_ms, BENCH_ITERS);
            printf("%-6d  %-10s  %10.3f  %7.3f  %8.3f  %9.3f  %9s\n",
                   N, "CPU", cpu.mean_ms, cpu.std_ms, cpu_gb, cpu_gf, "1.00x");
            if (csv_output)
                printf("BENCH_CSV,scalability,%d,CPU,%d,%d,%d,"
                       "%.4f,%.4f,%.4f,%.4f,1.0000,0.0000\n",
                       N, TILE_X, TILE_Y, num_blk,
                       cpu.mean_ms, cpu.std_ms, cpu_gb, cpu_gf);
        }

        // ── GPU V1 (naive) ────────────────────────────────────────────────────
        BenchStats v1 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 0);
        double v1_gb = gb_per_sec(N, v1.mean_ms, BENCH_ITERS);
        double v1_gf = gflops_per_sec(N, v1.mean_ms, BENCH_ITERS);
        double v1_sp = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v1.mean_ms : 0.0;
        printf("%-6d  %-10s  %10.3f  %7.3f  %8.3f  %9.3f  %8.2fx\n",
               N, "GPU V1", v1.mean_ms, v1.std_ms, v1_gb, v1_gf, v1_sp);
        if (csv_output)
            printf("BENCH_CSV,scalability,%d,V1,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,1.0000\n",
                   N, TILE_X, TILE_Y, num_blk,
                   v1.mean_ms, v1.std_ms, v1_gb, v1_gf, v1_sp);

        // ── GPU V2 (shared-mem) ────────────────────────────────────────────────
        BenchStats v2 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 1);
        double v2_gb    = gb_per_sec(N, v2.mean_ms, BENCH_ITERS);
        double v2_gf    = gflops_per_sec(N, v2.mean_ms, BENCH_ITERS);
        double v2_sp    = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v2.mean_ms : 0.0;
        double v2_vs_v1 = (v1.mean_ms > 0.0) ? v1.mean_ms / v2.mean_ms : 1.0;
        printf("%-6d  %-10s  %10.3f  %7.3f  %8.3f  %9.3f  %8.2fx  (%.2fx vs V1)\n",
               N, "GPU V2", v2.mean_ms, v2.std_ms, v2_gb, v2_gf, v2_sp, v2_vs_v1);
        if (csv_output)
            printf("BENCH_CSV,scalability,%d,V2,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   N, TILE_X, TILE_Y, num_blk,
                   v2.mean_ms, v2.std_ms, v2_gb, v2_gf, v2_sp, v2_vs_v1);

        // ── GPU V3 (coalesced) ─────────────────────────────────────────────────
        BenchStats v3 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 2);
        double v3_gb    = gb_per_sec(N, v3.mean_ms, BENCH_ITERS);
        double v3_gf    = gflops_per_sec(N, v3.mean_ms, BENCH_ITERS);
        double v3_sp    = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v3.mean_ms : 0.0;
        double v3_vs_v2 = (v2.mean_ms > 0.0) ? v2.mean_ms / v3.mean_ms : 1.0;
        printf("%-6d  %-10s  %10.3f  %7.3f  %8.3f  %9.3f  %8.2fx  (%.2fx vs V2)\n",
               N, "GPU V3", v3.mean_ms, v3.std_ms, v3_gb, v3_gf, v3_sp, v3_vs_v2);
        if (csv_output)
            printf("BENCH_CSV,scalability,%d,V3,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   N, TILE_X, TILE_Y, num_blk,
                   v3.mean_ms, v3.std_ms, v3_gb, v3_gf, v3_sp, v3_vs_v2);

        printf("---------------------------------------------------------\n");

        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_u_new));
        CUDA_CHECK(cudaFree(d_f));
        free(h_u); free(h_f);
    }
    printf("=========================================================\n\n");
}


void run_block_size_sweep(int N, bool csv_output) {
    // Block configurations to test
    struct Cfg { int bx, by; };
    static const Cfg cfgs[] = {
        { 8,  8},   //   64 threads/block
        {16,  8},   //  128 threads/block
        {16, 16},   //  256 threads/block  <- REFERENCE (TILE_X x TILE_Y)
        {32,  8},   //  256 threads/block
        {32, 16},   //  512 threads/block
        {32, 32},   // 1024 threads/block  (device max)
    };
    const int n_cfgs = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    printf("=========================================================\n");
    printf("  BLOCK-SIZE SWEEP  (Strong Scaling, fixed N=%d)\n", N);
    printf("  Kernel: jacobi_kernel_flex (V1 w/ dynamic blockDim)\n");
    printf("  Reference: 16x16 | Runs: %d | Iters/run: %d\n",
           N_BENCH_RUNS, BENCH_ITERS);
    printf("  Ref [3] Volkov (2010) GTC; [4] Kirk & Hwu (2016) Chap. 5\n");
    printf("=========================================================\n");
    printf("%-10s  %8s  %8s  %10s  %7s  %8s  %11s  %11s\n",
           "BlockCfg", "Thr/Blk", "NumBlks",
           "Mean(ms)", "Std(ms)", "GB/s",
           "Spd vs Ref", "Spd vs CPU");
    printf("---------------------------------------------------------\n");

    if (csv_output) {
        printf("# BENCH_CSV format: tag,N,bx,by,num_blocks,"
               "mean_ms,std_ms,gb_s,gflops,speedup_ref16x16,speedup_cpu\n");
    }

    const double h     = 1.0 / (N - 1);
    const size_t bytes = (size_t)N * N * sizeof(double);

    SolverParams params;
    params.N           = N;
    params.h           = h;
    params.max_iter    = BENCH_ITERS;
    params.check_every = BENCH_ITERS + 1;
    params.tol         = 0.0;

    double* h_u = (double*)malloc(bytes);
    double* h_f = (double*)malloc(bytes);
    initialize(h_u, h_f, N, h);

    double *d_u = nullptr, *d_u_new = nullptr, *d_f = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u,     bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_f,     bytes));
    CUDA_CHECK(cudaMemcpy(d_u,     h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,     h_f, bytes, cudaMemcpyHostToDevice));

    // CPU baseline (needed for speedup vs CPU column)
    BenchStats cpu = cpu_bench_avg(params, N_BENCH_RUNS, BENCH_ITERS);
    {
        double cpu_gb = gb_per_sec(N, cpu.mean_ms, BENCH_ITERS);
        printf("%-10s  %8d  %8s  %10.3f  %7.3f  %8.3f  %11s  %11s (baseline)\n",
               "CPU seq", 1, "-", cpu.mean_ms, cpu.std_ms, cpu_gb, "-", "1.00x");
    }

    // Pre-compute the 16x16 reference stats
    BenchStats ref16 = gpu_bench_avg_bs(params, d_u, d_u_new, d_f,
                                         N_BENCH_RUNS, BENCH_ITERS, TILE_X, TILE_Y);

    // Sweep all block configs using jacobi_kernel_flex
    for (int ci = 0; ci < n_cfgs; ++ci) {
        int bx = cfgs[ci].bx;
        int by = cfgs[ci].by;

        int thr     = bx * by;
        int num_blk = ((N + bx - 1) / bx) * ((N + by - 1) / by);

        BenchStats s = gpu_bench_avg_bs(params, d_u, d_u_new, d_f,
                                         N_BENCH_RUNS, BENCH_ITERS, bx, by);
        double gb   = gb_per_sec(N, s.mean_ms, BENCH_ITERS);
        double sp_r = (ref16.mean_ms > 0.0) ? ref16.mean_ms / s.mean_ms : 1.0;
        double sp_c = (cpu.mean_ms   > 0.0) ? cpu.mean_ms   / s.mean_ms : 0.0;

        char cfg_str[12];
        snprintf(cfg_str, sizeof(cfg_str), "%dx%d", bx, by);
        const char* mark = (bx == TILE_X && by == TILE_Y) ? "*" : " ";

        printf("%-10s%1s %8d  %8d  %10.3f  %7.3f  %8.3f  %10.3fx  %10.3fx\n",
               cfg_str, mark, thr, num_blk,
               s.mean_ms, s.std_ms, gb, sp_r, sp_c);

        if (csv_output)
            printf("BENCH_CSV,block_sweep,%d,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   N, bx, by, num_blk,
                   s.mean_ms, s.std_ms, gb,
                   gflops_per_sec(N, s.mean_ms, BENCH_ITERS), sp_r, sp_c);
    }

    // Also benchmark V2 and V3 at 16x16 for additional comparison curves (slide 13)
    printf("\n  --- Optimised kernels at 16x16 for comparison ---\n");
    int ref_blk = ((N + TILE_X - 1) / TILE_X) * ((N + TILE_Y - 1) / TILE_Y);
    {
        BenchStats v2 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 1);
        double gb   = gb_per_sec(N, v2.mean_ms, BENCH_ITERS);
        double sp_r = (ref16.mean_ms > 0.0) ? ref16.mean_ms / v2.mean_ms : 1.0;
        double sp_c = (cpu.mean_ms   > 0.0) ? cpu.mean_ms   / v2.mean_ms : 0.0;
        printf("  GPU V2 (shmem)  16x16 : mean=%.3f ms  GB/s=%.2f  "
               "sp_ref=%.3fx  sp_cpu=%.3fx\n",
               v2.mean_ms, gb, sp_r, sp_c);
        if (csv_output)
            printf("BENCH_CSV,block_sweep_v2,%d,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   N, TILE_X, TILE_Y, ref_blk,
                   v2.mean_ms, v2.std_ms, gb,
                   gflops_per_sec(N, v2.mean_ms, BENCH_ITERS), sp_r, sp_c);
    }
    {
        BenchStats v3 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 2);
        double gb   = gb_per_sec(N, v3.mean_ms, BENCH_ITERS);
        double sp_r = (ref16.mean_ms > 0.0) ? ref16.mean_ms / v3.mean_ms : 1.0;
        double sp_c = (cpu.mean_ms   > 0.0) ? cpu.mean_ms   / v3.mean_ms : 0.0;
        printf("  GPU V3 (coal.)  16x16 : mean=%.3f ms  GB/s=%.2f  "
               "sp_ref=%.3fx  sp_cpu=%.3fx\n",
               v3.mean_ms, gb, sp_r, sp_c);
        if (csv_output)
            printf("BENCH_CSV,block_sweep_v3,%d,%d,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   N, TILE_X, TILE_Y, ref_blk,
                   v3.mean_ms, v3.std_ms, gb,
                   gflops_per_sec(N, v3.mean_ms, BENCH_ITERS), sp_r, sp_c);
    }

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_new));
    CUDA_CHECK(cudaFree(d_f));
    free(h_u); free(h_f);

    printf("=========================================================\n\n");
}


void run_weak_scaling_benchmark(bool run_cpu, bool csv_output) {
    // Weak scaling: fix n = TILE_X*TILE_Y = 256 threads/block; vary b.
    // N = sqrt(n * b)  ->  N = 16 * sqrt(b)
    // b values and corresponding N (all multiples of TILE_X=16):
    //   b=4   -> N=32
    //   b=16  -> N=64
    //   b=64  -> N=128
    //   b=256 -> N=256
    //   b=1024-> N=512
    //   b=4096-> N=1024

    struct WPoint { int b; int N; };
    static const WPoint pts[] = {
        {    4,   32},
        {   16,   64},
        {   64,  128},
        {  256,  256},
        { 1024,  512},
        { 4096, 1024},
    };
    const int n_pts = (int)(sizeof(pts) / sizeof(pts[0]));
    const int n_threads_per_block = TILE_X * TILE_Y;   // n = 256

    printf("=========================================================\n");
    printf("  WEAK SCALING BENCHMARK\n");
    printf("  n = %d threads/block (fixed); N = 16*sqrt(b) grows with b\n",
           n_threads_per_block);
    printf("  SS(b,n) = T_seq(1,n*b) / T_version(b,n*b)  [slide 16]\n");
    printf("  Ideal: SS constant (= speedup at b=1)\n");
    printf("  Runs: %d | Iters/run: %d\n", N_BENCH_RUNS, BENCH_ITERS);
    printf("=========================================================\n");
    printf("%-6s  %-7s  %10s  %10s  %10s  %10s  %9s  %9s  %9s\n",
           "b", "N", "T_cpu(ms)", "T_V1(ms)", "T_V2(ms)", "T_V3(ms)",
           "SS_V1", "SS_V2", "SS_V3");
    printf("---------------------------------------------------------\n");

    if (csv_output) {
        printf("# BENCH_CSV format: tag,num_blocks,N,solver,"
               "mean_ms,std_ms,gb_s,scaled_speedup\n");
    }

    for (int pi = 0; pi < n_pts; ++pi) {
        int b = pts[pi].b;
        int N = pts[pi].N;

        const double h     = 1.0 / (N - 1);
        const size_t bytes = (size_t)N * N * sizeof(double);

        SolverParams params;
        params.N           = N;
        params.h           = h;
        params.max_iter    = BENCH_ITERS;
        params.check_every = BENCH_ITERS + 1;
        params.tol         = 0.0;

        double* h_u = (double*)malloc(bytes);
        double* h_f = (double*)malloc(bytes);
        initialize(h_u, h_f, N, h);

        double *d_u = nullptr, *d_u_new = nullptr, *d_f = nullptr;
        CUDA_CHECK(cudaMalloc(&d_u,     bytes));
        CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
        CUDA_CHECK(cudaMalloc(&d_f,     bytes));
        CUDA_CHECK(cudaMemcpy(d_u,     h_u, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_new, h_u, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,     h_f, bytes, cudaMemcpyHostToDevice));

        // CPU T_seq(N)
        BenchStats cpu = { -1.0, 0.0 };
        if (run_cpu) {
            int cpu_runs = (N > 512) ? N_CPU_RUNS_LARGE : N_BENCH_RUNS;
            cpu = cpu_bench_avg(params, cpu_runs, BENCH_ITERS);
        }

        // GPU V1, V2, V3
        BenchStats v1 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 0);
        BenchStats v2 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 1);
        BenchStats v3 = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                       N_BENCH_RUNS, BENCH_ITERS, 2);

        // Scaled speedup SS = T_cpu / T_gpu
        double ss_v1 = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v1.mean_ms : 0.0;
        double ss_v2 = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v2.mean_ms : 0.0;
        double ss_v3 = (run_cpu && cpu.mean_ms > 0.0) ? cpu.mean_ms / v3.mean_ms : 0.0;

        if (run_cpu)
            printf("%-6d  %-7d  %10.3f  %10.3f  %10.3f  %10.3f  "
                   "%9.3f  %9.3f  %9.3f\n",
                   b, N, cpu.mean_ms, v1.mean_ms, v2.mean_ms, v3.mean_ms,
                   ss_v1, ss_v2, ss_v3);
        else
            printf("%-6d  %-7d  %10s  %10.3f  %10.3f  %10.3f  "
                   "%9s  %9s  %9s\n",
                   b, N, "skipped", v1.mean_ms, v2.mean_ms, v3.mean_ms,
                   "-", "-", "-");

        if (csv_output) {
            double v1_gb = gb_per_sec(N, v1.mean_ms, BENCH_ITERS);
            double v2_gb = gb_per_sec(N, v2.mean_ms, BENCH_ITERS);
            double v3_gb = gb_per_sec(N, v3.mean_ms, BENCH_ITERS);
            if (run_cpu && cpu.mean_ms > 0.0) {
                double cpu_gb = gb_per_sec(N, cpu.mean_ms, BENCH_ITERS);
                printf("BENCH_CSV,weak_scaling,%d,%d,CPU,%.4f,%.4f,%.4f,1.0000\n",
                       b, N, cpu.mean_ms, cpu.std_ms, cpu_gb);
            }
            printf("BENCH_CSV,weak_scaling,%d,%d,V1,%.4f,%.4f,%.4f,%.4f\n",
                   b, N, v1.mean_ms, v1.std_ms, v1_gb, ss_v1);
            printf("BENCH_CSV,weak_scaling,%d,%d,V2,%.4f,%.4f,%.4f,%.4f\n",
                   b, N, v2.mean_ms, v2.std_ms, v2_gb, ss_v2);
            printf("BENCH_CSV,weak_scaling,%d,%d,V3,%.4f,%.4f,%.4f,%.4f\n",
                   b, N, v3.mean_ms, v3.std_ms, v3_gb, ss_v3);
        }

        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_u_new));
        CUDA_CHECK(cudaFree(d_f));
        free(h_u); free(h_f);
    }
    printf("=========================================================\n\n");
}


void run_communication_overhead(int N, bool csv_output) {
    const double h      = 1.0 / (N - 1);
    const size_t bytes  = (size_t)N * N * sizeof(double);
    const int check_ev  = DEFAULT_CHECK_EVERY;   // = 100
    const int num_blk   = ((N + TILE_X - 1) / TILE_X) * ((N + TILE_Y - 1) / TILE_Y);

    printf("=========================================================\n");
    printf("  COMMUNICATION OVERHEAD ANALYSIS  (N=%d)\n", N);
    printf("  check_every = %d iterations\n", check_ev);
    printf("  Strategy A (V1): copy 2*N^2*8 = %.2f MB per check (D->H)\n",
           2.0 * N * N * 8.0 / (1024.0 * 1024.0));
    printf("  Strategy B (V2): copy num_blocks*8 = %.2f KB per check (D->H)\n",
           (double)num_blk * 8.0 / 1024.0);
    printf("=========================================================\n");

    // Allocate host and device buffers
    double* h_u = (double*)malloc(bytes);
    double* h_f = (double*)malloc(bytes);
    double* h_buf_a = (double*)malloc(2 * bytes);       // Strategy A host buf
    double* h_buf_b = (double*)malloc((size_t)num_blk * sizeof(double));
    if (!h_u || !h_f || !h_buf_a || !h_buf_b) {
        fprintf(stderr, "[comm_overhead] malloc failed\n"); exit(EXIT_FAILURE);
    }
    initialize(h_u, h_f, N, h);

    double *d_u = nullptr, *d_u_new = nullptr, *d_f = nullptr, *d_blkmax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u,     bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_f,     bytes));
    CUDA_CHECK(cudaMalloc(&d_blkmax, (size_t)num_blk * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_u,     h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_new, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,     h_f, bytes, cudaMemcpyHostToDevice));

    SolverParams params;
    params.N           = N;
    params.h           = h;
    params.max_iter    = BENCH_ITERS;
    params.check_every = BENCH_ITERS + 1;
    params.tol         = 0.0;

    // ── 1. Kernel-only time per iteration ────────────────────────────────────
    // Average over N_BENCH_RUNS runs of BENCH_ITERS iterations
    BenchStats v1_kern = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                        N_BENCH_RUNS, BENCH_ITERS, 0);
    BenchStats v2_kern = gpu_bench_avg(params, d_u, d_u_new, d_f,
                                        N_BENCH_RUNS, BENCH_ITERS, 1);
    double v1_ms_iter = v1_kern.mean_ms / BENCH_ITERS;
    double v2_ms_iter = v2_kern.mean_ms / BENCH_ITERS;

    // ── 2. Strategy A: time a single D->H copy of 2*N^2 doubles ─────────────
    CUDA_CHECK(cudaDeviceSynchronize());   // ensure all pending GPU work is done
    double sum_a = 0.0;
    for (int r = 0; r < N_BENCH_RUNS; ++r) {
        CpuTimer t; t.start();
        // Simulate the V1 convergence check: copy both u and u_new from device
        CUDA_CHECK(cudaMemcpy(h_buf_a,           d_u,     bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_buf_a + N * N,   d_u_new, bytes, cudaMemcpyDeviceToHost));
        t.stop();
        sum_a += (double)t.elapsed_ms();
    }
    double transfer_a_ms = sum_a / N_BENCH_RUNS;  // time for one full Strategy A check

    // ── 3. Strategy B: time a single D->H copy of num_blocks doubles ─────────
    double sum_b = 0.0;
    for (int r = 0; r < N_BENCH_RUNS; ++r) {
        CpuTimer t; t.start();
        CUDA_CHECK(cudaMemcpy(h_buf_b, d_blkmax,
                              (size_t)num_blk * sizeof(double),
                              cudaMemcpyDeviceToHost));
        t.stop();
        sum_b += (double)t.elapsed_ms();
    }
    double transfer_b_ms = sum_b / N_BENCH_RUNS;  // time for one full Strategy B check

    // ── 4. Overhead calculations ──────────────────────────────────────────────
    // Transfer overhead per iteration = transfer_per_check / check_every
    double overhead_a_per_iter = transfer_a_ms / (double)check_ev;
    double overhead_b_per_iter = transfer_b_ms / (double)check_ev;

    // Overhead fraction relative to kernel time
    double pct_a_v1 = (v1_ms_iter > 0.0) ? 100.0 * overhead_a_per_iter / v1_ms_iter : 0.0;
    double pct_b_v2 = (v2_ms_iter > 0.0) ? 100.0 * overhead_b_per_iter / v2_ms_iter : 0.0;

    printf("\n  %-40s %12s\n", "Metric", "Value");
    printf("  %-40s %12.4f ms\n", "V1 kernel time per iteration",   v1_ms_iter);
    printf("  %-40s %12.4f ms\n", "V2 kernel time per iteration",   v2_ms_iter);
    printf("\n");
    printf("  %-40s %12.4f ms\n", "Strategy A: full D->H copy",   transfer_a_ms);
    printf("    (2 * N^2 * 8 = %.2f MB  at  %.1f GB/s effective)\n",
           2.0 * N * N * 8.0 / (1024.0 * 1024.0),
           (2.0 * N * N * 8.0 / 1e9) / (transfer_a_ms / 1000.0));
    printf("  %-40s %12.4f ms\n", "Strategy A overhead per iter",  overhead_a_per_iter);
    printf("  %-40s %11.2f%%\n",  "Strategy A overhead % (vs V1)", pct_a_v1);
    printf("\n");
    printf("  %-40s %12.4f ms\n", "Strategy B: block-max D->H copy", transfer_b_ms);
    printf("    (%d blocks * 8 = %.2f KB  at  %.1f GB/s effective)\n",
           num_blk, (double)num_blk * 8.0 / 1024.0,
           ((double)num_blk * 8.0 / 1e9) / (transfer_b_ms / 1000.0));
    printf("  %-40s %12.4f ms\n", "Strategy B overhead per iter",  overhead_b_per_iter);
    printf("  %-40s %11.2f%%\n",  "Strategy B overhead % (vs V2)", pct_b_v2);
    printf("\n");
    printf("  Transfer reduction A->B : %.1fx  (%.2f MB -> %.2f KB)\n",
           transfer_a_ms / transfer_b_ms,
           2.0 * N * N * 8.0 / (1024.0 * 1024.0),
           (double)num_blk * 8.0 / 1024.0);

    if (csv_output) {
        printf("BENCH_CSV,comm_overhead,%d,StratA,kernel_per_iter,"
               "%.6f,0.000000,0.000000,0.000000\n", N, v1_ms_iter);
        printf("BENCH_CSV,comm_overhead,%d,StratA,transfer_per_check,"
               "%.6f,0.000000,%.4f,%.4f\n", N, transfer_a_ms,
               2.0 * N * N * 8.0 / (1024.0 * 1024.0), pct_a_v1);
        printf("BENCH_CSV,comm_overhead,%d,StratB,kernel_per_iter,"
               "%.6f,0.000000,0.000000,0.000000\n", N, v2_ms_iter);
        printf("BENCH_CSV,comm_overhead,%d,StratB,transfer_per_check,"
               "%.6f,0.000000,%.6f,%.4f\n", N, transfer_b_ms,
               (double)num_blk * 8.0 / 1024.0, pct_b_v2);
    }

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_new));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_blkmax));
    free(h_u); free(h_f); free(h_buf_a); free(h_buf_b);

    printf("=========================================================\n\n");
}
/**
 * benchmark.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Benchmark Suite for the Jacobi Poisson mini-solver.
 * Covers all measurement requirements:
 *
 *  Slide  7 │ print_roofline_analysis    – FLOP/byte, AI, memory/compute-bound
 *  Slide 12 │ run_scalability_benchmark  – throughput vs N, mean ± σ
 *  Slide 13 │ run_block_size_sweep       – strong-scaling config sweep
 *  Slide 15 │ run_weak_scaling_benchmark – SS(b,n) weak-scaling study
 *  Slide 19 │ run_communication_overhead – H↔D transfer vs kernel analysis
 *
 * Statistics: every timed result is the mean ± σ over N_BENCH_RUNS independent
 * runs, satisfying the ≥ 20 runs requirement
 * ─────────────────────────────────────────────────────────────────────────────
 */

#pragma once

#include <vector>

#define N_BENCH_RUNS       20
#define BENCH_ITERS        200
#define N_CPU_RUNS_LARGE    5

struct BenchStats {
    double mean_ms;
    double std_ms;
};

/**
 * print_roofline_analysis
 *
 * Prints FLOP/byte estimates, arithmetic intensity, and roofline analysis
 * for NVIDIA T4 GPU (sm_75, 320 GB/s, 0.260 TFLOPS FP64).
 *
 * For the 5-point Jacobi stencil:
 *   FLOPs/point = 6  (4 adds, 1 multiply h²·f, 1 multiply x0.25)
 *   Streaming lower bound: 3 arrays × 8 B = 24 B/point  (2 reads + 1 write)
 *   V1 AI = 6/24 = 0.25 FLOP/B   <<  T4 ridge ≈ 0.81 FLOP/B  --> MEMORY-BOUND
 *
 * References
 * ──────────
 * [1] Williams, S., Waterman, A., Patterson, D. (2009). "Roofline: An Insightful
 *     Visual Performance Model for Multicore Architectures." CACM 52(4), 65-76.
 * [2] NVIDIA. (2023). "NVIDIA T4 GPU Datasheet."
 */
void print_roofline_analysis();

/**
 * run_scalability_benchmark
 *
 * Run N_BENCH_RUNS independent timed experiments for each grid size in `sizes`
 * with fixed block configuration TILE_X x TILE_Y = 16x16.
 * Reports mean time [ms], σ, effective GB/s, GFLOPs/s, speedup vs CPU.
 *
 * Bandwidth formula (streaming lower bound):
 *   BW = 3 × N² × 8 bytes / t_s   [Williams et al. 2009]
 */
void run_scalability_benchmark(const std::vector<int>& sizes, bool run_cpu, bool csv_output = false);

/**
 * run_block_size_sweep
 *
 * Strong-scaling study: fixed grid size N, vary thread-block configuration.
 * Configurations: {8x8, 16x8, 16x16(*), 32x8, 32x16, 32x32}  (*) = reference
 *
 * Uses jacobi_kernel_flex (V1 variant, blockDim-indexed) for arbitrary configs.
 * Reports time [ms], speedup vs 16x16 reference, speedup vs CPU.
 * Also runs V2 and V3 at 16x16 for comparison (added curves, slide 13).
 *
 * References
 * ──────────
 * [3] Volkov, V. (2010). "Better Performance at Lower Occupancy."
 *     GPU Technology Conference, NVIDIA.
 * [4] Kirk, D. & Hwu, W. (2016). "Programming Massively Parallel Processors,"
 *     3rd ed., Chapter 5. Elsevier.
 */
void run_block_size_sweep(int N, bool csv_output = false);

/**
 * run_weak_scaling_benchmark
 *
 * Weak-scaling study: problem size N grows proportionally to resources.
 * Fix n = TILE_X × TILE_Y = 256 threads/block; sweep b = {4, 16, 64, 256, 1024, 4096}
 * so that N = sqrt(n × b) in {32, 64, 128, 256, 512, 1024}.
 *
 * Scaled speedup (Lezione 16_0, slide 16):
 *   SS(b, n) = T_seq(1, n*b) / T_versione(b, n*b)
 * Ideal: SS ≈ constant across all b.
 */
void run_weak_scaling_benchmark(bool run_cpu, bool csv_output = false);

/**
 * run_communication_overhead
 *
 * Measures H↔D transfer overhead against kernel compute time.
 *
 * Strategy A (V1 naive): every check_every iters copies 2*N²*8 bytes D->H.
 * Strategy B (V2/V3):    every check_every iters copies num_blocks*8 bytes D->H.
 *
 * Reports kernel time per iter, transfer time per check, overhead %.
 */
void run_communication_overhead(int N, bool csv_output = false);
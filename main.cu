/** [COLAB]
!nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude                 \
      src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu    \
      src/validation.cpp src/gpu_utils.cu src/solver_runner.cu \
      src/benchmark.cu main.cu -o minisolver

!nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude                 \
      src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu    \
      src/validation.cpp src/gpu_utils.cu src/solver_runner.cu \
      src/benchmark.cu tests/test_suite.cu -o test_runner

*/

// Esecuzione normale:  !./minisolver --n 256
// Test di correttezza: !./test_runner

/** [ESECUZIONE CON RACCOLTA DATI PER I GRAFICI]

!./minisolver --n 512 --csv 2>/dev/null | grep BENCH_CSV > results.csv
!pip install matplotlib -q
!python3 analyze_results.py results.csv

*/

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "include/common.h"
#include "include/benchmark.h"
#include "include/gpu_utils.h"
#include "include/solver_runner.h"

// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Solver options:\n");
    printf("  --n <int>              Grid size N (default: 256; auto-aligned to 16)\n");
    printf("  --tol <double>         Convergence tolerance (default: 1e-7)\n");
    printf("  --max-iter <int>       Max Jacobi iterations (default: 50000)\n");
    printf("  --check-every <int>    Iters between convergence checks (default: 100)\n");
    printf("\nBenchmark control:\n");
    printf("  --no-scalability       Skip the scalability benchmark\n");
    printf("  --no-block-sweep       Skip the block-size sweep (strong scaling)\n");
    printf("  --no-weak-scaling      Skip the weak scaling benchmark\n");
    printf("  --no-comm-overhead     Skip the communication overhead analysis\n");
    printf("  --block-sweep-n <int>  Grid size N used for block-size sweep (default: 512)\n");
    printf("  --comm-n <int>         Grid size N used for comm-overhead analysis (default: 512)\n");
    printf("\nOutput options:\n");
    printf("  --no-cpu               Skip the sequential CPU execution\n");
    printf("  --csv                  Emit BENCH_CSV,... lines parseable by analyze_results.py\n");
    printf("  -h, --help             Show this help message\n");

    printf("\nExample (collect CSV for plotting):\n");
    printf("  ./minisolver --n 512 --csv 2>/dev/null | grep BENCH_CSV > results.csv\n");
    printf("  python3 analyze_results.py results.csv\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {

    // ── Default parameters ────────────────────────────────────────────────────
    int    N              = 256;
    double tol            = 1.0e-7;
    int    max_iter       = 50000;
    int    check_every    = DEFAULT_CHECK_EVERY;   // from common.h (= 100)

    bool   run_cpu           = true;
    bool   csv_output        = false;
    bool   run_scalability   = true;
    bool   run_block_sweep   = true;
    bool   run_weak_scaling  = true;
    bool   run_comm_overhead = true;
    int    block_sweep_n     = 512;
    int    comm_n            = 512;

    // ── Command-line parsing ──────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--n" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        }
        else if (arg == "--tol" && i + 1 < argc) {
            tol = std::atof(argv[++i]);
        }
        else if (arg == "--max-iter" && i + 1 < argc) {
            max_iter = std::atoi(argv[++i]);
        }
        else if (arg == "--check-every" && i + 1 < argc) {
            check_every = std::atoi(argv[++i]);
        }
        else if (arg == "--no-cpu") {
            run_cpu = false;
        }
        else if (arg == "--csv") {
            csv_output = true;
        }
        else if (arg == "--no-scalability") {
            run_scalability = false;
        }
        else if (arg == "--no-block-sweep") {
            run_block_sweep = false;
        }
        else if (arg == "--no-weak-scaling") {
            run_weak_scaling = false;
        }
        else if (arg == "--no-comm-overhead") {
            run_comm_overhead = false;
        }
        else if (arg == "--block-sweep-n" && i + 1 < argc) {
            block_sweep_n = std::atoi(argv[++i]);
        }
        else if (arg == "--comm-n" && i + 1 < argc) {
            comm_n = std::atoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "[WARNING] Unknown option: %s  (run with --help)\n",
                    argv[i]);
        }
    }

    // ── Align grid sizes to tile dimensions ───────────────────────────────────
    align_grid_size(N);
    align_grid_size(block_sweep_n);
    align_grid_size(comm_n);

    // ── Build solver parameters ───────────────────────────────────────────────
    const double h = 1.0 / (N - 1);

    SolverParams params;
    params.N           = N;
    params.max_iter    = max_iter;
    params.check_every = check_every;
    params.h           = h;
    params.tol         = tol;

    print_gpu_properties();

    test_standalone_reduction();

    print_roofline_analysis();

    printf("=========================================================\n");
    printf("  FULL JACOBI POISSON SOLVE  (N = %d, tol = %.1e)\n", N, tol);
    printf("=========================================================\n");

    FullSolverResults results = run_full_solve(params, run_cpu);
    print_solver_results(results, params);

    if (run_scalability) {
        // Standard grid sizes spanning one decade of problem size
        std::vector<int> sizes = {128, 256, 512, 1024};
        run_scalability_benchmark(sizes, run_cpu, csv_output);
    }

    // ── Block-size sweep / strong-scaling
    //    Fixes N = block_sweep_n, sweeps {8x8, 16x8, 16x16, 32x8, 32x16, 32x32}.
    //    x-axis: num_blocks;  y-axis: time [ms] and speedup.
    if (run_block_sweep) {
        run_block_size_sweep(block_sweep_n, csv_output);
    }

    // ── Weak scaling: SS(b,n) study
    //    Fix n=256 threads/block, grow b so N = 16*sqrt(b) ∈ {32, 64, ... , 1024}.
    //    Ideal: SS ≈ constant (time per iter ≈ constant).
    if (run_weak_scaling) {
        run_weak_scaling_benchmark(run_cpu, csv_output);
    }

    // ── Communication overhead
    //    Strategy A (V1)   : copies 2*N^2*8 bytes per check.
    //    Strategy B (V2/V3): copies num_blocks*8 bytes per check.
    if (run_comm_overhead) {
        run_communication_overhead(comm_n, csv_output);
    }

    return 0;
}
// Compile:
// !nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu src/validation.cpp src/gpu_utils.cu src/solver_runner.cu src/benchmark.cu main.cu -o minisolver

// Execute (need to set parameters):
// !./minisolver --n 256 --tol 1e-7

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "include/common.h"
#include "include/benchmark.h"
#include "include/gpu_utils.h"
#include "include/solver_runner.h"

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

    // Align grid size to block/tile size
    align_grid_size(N);

    double h = 1.0 / (N - 1);
    SolverParams params = { N, h, tol, max_iter, check_every };

    print_gpu_properties();

    test_standalone_reduction();

    printf("=========================================================\n");
    printf("  FULL JACOBI POISSON SOLVE (N = %d, tol = %.1e)\n", N, tol);
    printf("=========================================================\n");

    FullSolverResults results = run_full_solve(params, run_cpu);

    print_solver_results(results, params);

    // [LAST] Scalability Benchmark Suite
    if (run_scalability) {
        std::vector<int> sizes = {128, 256, 512, 1024};
        run_scalability_benchmark(sizes, run_cpu);
    }

    return 0;
}
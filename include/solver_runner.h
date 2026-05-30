#pragma once

#include "common.h"

// Struct to store results from CPU and all GPU variants
struct FullSolverResults {
    bool run_cpu;
    SolverResult cpu_res;
    SolverResult v1_res;
    SolverResult v2_res;
    SolverResult v3_res;

    double cpu_analytical_err;
    double v1_analytical_err;
    double v2_analytical_err;
    double v3_analytical_err;

    double v1_vs_cpu_diff;
    double v2_vs_cpu_diff;
    double v3_vs_cpu_diff;
};

// Aligns the grid size N to be a multiple of block tile size (TILE_X, TILE_Y)
void align_grid_size(int& N);

// Runs the Jacobi solve on CPU and all GPU versions.
// Allocates/frees memory internally.
FullSolverResults run_full_solve(const SolverParams& params, bool run_cpu);

// Formats and prints the solver results and validation report to stdout.
void print_solver_results(const FullSolverResults& results, const SolverParams& params);

// Runs a standalone test of the GPU max-reduction kernel.
// Returns true if the test passes, false otherwise.
bool test_standalone_reduction();
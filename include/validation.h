#pragma once

#include "common.h"

/**
 * Returns max_{k} |a[k] - b[k]|  over n elements.
 */
double max_abs_diff(const double* a, const double* b, int n);

/**
 * Returns sqrt( sum_{k} (a[k]-b[k])^2 / n )  — the RMS difference.
 */
double rms_diff(const double* a, const double* b, int n);

/**
 * Exact analytical solution at grid point (i, j).
 * u_exact(i,j) = sin(π · j·h) · sin(π · i·h)
 */
double exact_at(int i, int j, double h);

/**
 * Computes max |u[i,j] - u_exact(i,j)| over all N×N grid points.
 * Boundary points are included (they should all be ≈ 0).
 */
double max_error_vs_exact(const double* u, int N, double h);


void print_validation(
    const double* ref_sol,
    const double* test_sol,
    int           N,
    double        tol,
    const char*   label);
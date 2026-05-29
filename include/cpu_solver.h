/**
 * cpu_solver.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Serial (CPU) Jacobi solver for the 2-D Poisson equation.
 *
 * Problem:  −Δu = f  on [0,1]², u = 0 on ∂Ω
 * Source:   f(x,y) = 2π² sin(πx) sin(πy)
 * Exact:    u(x,y) = sin(πx) sin(πy)   --> used for validation
 *
 * Discretisation: 5-point finite-difference stencil on an N×N grid.
 * Jacobi update:
 *   u_new[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]
 *                         + h² · f[i,j])
 *
 * All arrays are flat, row-major:  element (i,j) is at index (i*N + j).
 * ─────────────────────────────────────────────────────────────────────────────
 */

#pragma once

#include "common.h"

/**
 * Initialize the solution and right-hand-side arrays.
 *
 * @param u   N×N flat array; set to 0 everywhere (boundary + interior)
 * @param f   N×N flat array; set to 2π²sin(πx)sin(πy) at each grid point
 * @param N   Grid size
 * @param h   Grid spacing  h = 1/(N-1)
 */
void initialize(double* u, double* f, int N, double h);

/**
 * Perform one Jacobi sweep over all interior points.
 *
 * @param u_new  Output: updated solution values
 * @param u      Input:  old solution values
 * @param f      Right-hand side
 * @param N      Grid size
 * @param h2     h² = (grid spacing)²
 * @return       max_{i,j} |u_new[i,j] − u[i,j]| over interior points
 */
double jacobi_step_cpu(
    double*       u_new,
    const double* u,
    const double* f,
    int           N,
    double        h2
);

/**
 * Run the full Jacobi iteration loop on the CPU.
 *
 * @param params  Solver params config
 * @param u       On entry: initial guess (all zeros). On exit: final solution
 * @param u_new   Scratch buffer of the same size as u
 * @param f       Right-hand side (unchanged on exit)
 * @return        SolverResult with iteration count, final error, and timing
 */
SolverResult jacobi_cpu(
    SolverParams  params,
    double*       u,
    double*       u_new,
    const double* f
);
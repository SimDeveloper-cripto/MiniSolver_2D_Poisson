#include <cmath>
#include <cstdio>
#include <cstring>

#include "../include/timer.h"
#include "../include/cpu_solver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void initialize(double* u, double* f, int N, double h) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double x = j * h;   // x corresponds to column
            const double y = i * h;   // y corresponds to row

            u[IDX(i, j, N)] = 0.0;
            f[IDX(i, j, N)] = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
}

double jacobi_step_cpu(double* u_new, const double* u, const double* f, int N, double h2) {
    double error = 0.0;

    // Iterate only over interior points
    // Skip Boundary row/col 0 and N-1
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            const int id = IDX(i, j, N);

            // 5-point Jacobi update
            const double val = 0.25 * (
                u[IDX(i - 1, j, N)] +   // top
                u[IDX(i + 1, j, N)] +   // bottom
                u[IDX(i, j - 1, N)] +   // left
                u[IDX(i, j + 1, N)] +   // right
                h2 * f[id]
            );
            u_new[id] = val;

            // Track what is the max change
            // --> For convergence criterion
            const double diff = std::fabs(val - u[id]);
            if (diff > error) error = diff;
        }
    }
    return error;
}

SolverResult jacobi_cpu(SolverParams  params, double* u, double* u_new, const double* f) {
    const int    N    = params.N;
    const double h2   = params.h * params.h;
    const double tol  = params.tol;
    const int    maxi = params.max_iter;

    CpuTimer timer;
    timer.start();

    double error = 1.0e30;
    int    iter  = 0;

    // Main Jacobi loop
    // Convention: u holds the current solution; u_new is the scratch buffer
    // After each sweep the two pointers are swapped (no data copy needed)
    while (iter < maxi) {
        ++iter;
        error = jacobi_step_cpu(u_new, u, f, N, h2);

        // Swap pointers: u_new becomes the new current solution
        double* tmp = u_new;
        u_new = u;
        u     = tmp;

        // Print progress
        if (iter % 500 == 0) {
            printf("[CPU]  iter = %6d   error = %.6e\n", iter, error);
        }

        if (error < tol) break;
    }

    timer.stop();

    // After the loop, the current solution is in u
    // Must use the returned arrays carefully. In main.cu we use device copies, so this is safe
    SolverResult res;
    res.iters       = iter;
    res.final_error = error;
    res.total_ms    = timer.elapsed_ms();
    res.ms_per_iter = (iter > 0) ? (res.total_ms / iter) : 0.0;
    return res;
}
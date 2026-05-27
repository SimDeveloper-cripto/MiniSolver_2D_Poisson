#include <cmath>
#include <cstdio>

#include "../include/validation.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double max_abs_diff(const double* a, const double* b, int n) {
    double mx = 0.0;
    for (int k = 0; k < n; ++k) {
        const double d = std::fabs(a[k] - b[k]);
        if (d > mx) mx = d;
    }
    return mx;
}

double rms_diff(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        const double d = a[k] - b[k];
        sum += d * d;
    }
    return (n > 0) ? std::sqrt(sum / n) : 0.0;
}

double exact_at(int i, int j, double h) {
    // u_exact(x, y) = sin(π x) sin(π y)
    // x = j·h
    // y = i·h
    return std::sin(M_PI * j * h) * std::sin(M_PI * i * h);
}

double max_error_vs_exact(const double* u, int N, double h) {
    double mx = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double diff = std::fabs(u[IDX(i, j, N)] - exact_at(i, j, h));
            if (diff > mx) mx = diff;
        }
    }
    return mx;
}

void print_validation(const double* ref_sol, const double* test_sol,
    int N, double tol, const char* label) {

    const int    n        = N * N;
    const double max_diff = max_abs_diff(ref_sol, test_sol, n);
    const double rms      = rms_diff(ref_sol, test_sol, n);
    const bool   pass     = (max_diff < tol * 100.0);

    printf("  %-18s  max|diff|=%.3e  rms=%.3e  %s\n",
        label, max_diff, rms, pass ? "[ PASS ]" : "[ FAIL ]");
}
#include <cstdio>
#include <cstdlib>

#include <cmath>
#include <string>

#include "../include/common.h"
#include "../include/solver_runner.h"

#define ASSERT_TRUE(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "  [FAIL] %s:%d: %s\n", __FILE__, __LINE__, message); \
            return false; \
        } \
    } while(0)

bool test_reduction_correctness() {
    printf("[TEST] Running standalone GPU reduction test...\n");
    bool ok = test_standalone_reduction();
    ASSERT_TRUE(ok, "Standalone GPU reduction test failed.");
    printf("  [PASS] Standalone GPU reduction is correct.\n\n");
    return true;
}

bool test_align_grid_size() {
    printf("[TEST] Running grid size alignment test...\n");
    
    int n1 = 250;
    align_grid_size(n1);
    ASSERT_TRUE(n1 == 256, "align_grid_size failed to round 250 to 256");

    int n2 = 256;
    align_grid_size(n2);
    ASSERT_TRUE(n2 == 256, "align_grid_size incorrectly modified an already aligned N = 256");

    int n3 = 1;
    align_grid_size(n3);
    ASSERT_TRUE(n3 == 16, "align_grid_size failed to round 1 to 16");

    printf("  [PASS] Grid size alignment logic is correct.\n\n");
    return true;
}

bool test_solvers_accuracy() {
    printf("[TEST] Running solver accuracy and validation test...\n");

    // Setup a small grid (128x128) with convergence tolerance 1e-6
    const int N      = 128;
    const double tol = 1e-6;
    const double h   = 1.0 / (N - 1);

    SolverParams params;
    params.N           = N;
    params.max_iter    = 20000;
    params.check_every = 100;
    params.h           = h;
    params.tol         = tol;

    FullSolverResults results = run_full_solve(params, true);

    // (should be > 0 and <= max_iter)
    ASSERT_TRUE(results.cpu_res.iters > 0, "CPU solver did not execute any iterations");
    ASSERT_TRUE(results.v1_res.iters  > 0, "GPU V1 solver did not execute any iterations");
    ASSERT_TRUE(results.v2_res.iters  > 0, "GPU V2 solver did not execute any iterations");
    ASSERT_TRUE(results.v3_res.iters  > 0, "GPU V3 solver did not execute any iterations");

    ASSERT_TRUE(std::abs(results.v1_res.iters - results.cpu_res.iters) <= params.check_every, "GPU V1 iters wildly different from CPU");
    ASSERT_TRUE(std::abs(results.v2_res.iters - results.cpu_res.iters) <= params.check_every, "GPU V2 iters wildly different from CPU");
    ASSERT_TRUE(std::abs(results.v3_res.iters - results.cpu_res.iters) <= params.check_every, "GPU V3 iters wildly different from CPU");

    ASSERT_TRUE(results.v1_vs_cpu_max_diff < tol * 100.0, "GPU V1 vs CPU max difference exceeds tolerance limit");
    ASSERT_TRUE(results.v2_vs_cpu_max_diff < tol * 100.0, "GPU V2 vs CPU max difference exceeds tolerance limit");
    ASSERT_TRUE(results.v3_vs_cpu_max_diff < tol * 100.0, "GPU V3 vs CPU max difference exceeds tolerance limit");

    ASSERT_TRUE(results.cpu_analytical_err < 1e-2, "CPU solution vs Exact is too large");
    ASSERT_TRUE(results.v1_analytical_err  < 1e-2, "GPU V1 solution vs Exact is too large");
    ASSERT_TRUE(results.v2_analytical_err  < 1e-2, "GPU V2 solution vs Exact is too large");
    ASSERT_TRUE(results.v3_analytical_err  < 1e-2, "GPU V3 solution vs Exact is too large");

    printf("  [PASS] Solvers accuracy and validation checks passed.\n\n");
    return true;
}

int main() {
    printf("=========================================================\n");
    printf("  MINISOLVER AUTOMATED TEST SUITE\n");
    printf("=========================================================\n\n");

    bool success = true;

    success &= test_reduction_correctness();
    success &= test_align_grid_size();
    success &= test_solvers_accuracy();

    printf("=========================================================\n");
    if (success) {
        printf("  ALL TESTS PASSED SUCCESSFULLY! [ PASS ]\n");
        printf("=========================================================\n");
        return EXIT_SUCCESS;
    } else {
        printf("  SOME TESTS FAILED! [ FAIL ]\n");
        printf("=========================================================\n");
        return EXIT_FAILURE;
    }
}
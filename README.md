# Overview

You can run this project using Google Colab, here's the tutorial !!

```sh
!nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude                 \
      src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu    \
      src/validation.cpp src/gpu_utils.cu src/solver_runner.cu \
      src/benchmark.cu main.cu -o minisolver

!nvcc -O3 -arch=sm_75 -use_fast_math -Iinclude                 \
      src/cpu_solver.cpp src/gpu_solver.cu src/reduction.cu    \
      src/validation.cpp src/gpu_utils.cu src/solver_runner.cu \
      src/benchmark.cu tests/test_suite.cu -o test_runner

!./minisolver --n 256

# Run:
# 1. Correctness Checks
# 2. Benchmarks
# 3. Tests
!./test_runner

# [DATA HARVESTING FOR GRAPHS AND STATISTICS]

!./minisolver --n 512 --csv 2>/dev/null | grep BENCH_CSV > results.csv
!pip install matplotlib -q
!python3 analyze_results.py results.csv
```
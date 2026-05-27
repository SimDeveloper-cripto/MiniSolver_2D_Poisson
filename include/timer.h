#pragma once

#include <chrono>

class CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TP    = Clock::time_point;
public:
    void   start()       { t0_ = Clock::now(); }
    void   stop()        { t1_ = Clock::now(); }
    double elapsed_ms()  const {
        return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
    }
private:
    TP t0_, t1_;
};

// ── CUDA-event-based timer (nvcc only)
#ifdef __CUDACC__

#include "common.h"
#include <cuda_runtime.h>

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#endif // __CUDACC__
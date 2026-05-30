#include <cstdio>
#include <cuda_runtime.h>

#include "../include/common.h"
#include "../include/gpu_utils.h"

void print_gpu_properties() {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        printf("No CUDA-capable GPU detected or CUDA runtime error.\n");
        return;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("=========================================================\n");
    printf("  GPU DEVICE PROPERTIES\n");
    printf("=========================================================\n");
    printf("  Device Name          : %s\n", prop.name);
    printf("  Compute Capability   : %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory        : %.2f GB\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Mem / Block   : %.2f KB\n", (double)prop.sharedMemPerBlock / 1024.0);
    printf("  Warp Size            : %d\n", prop.warpSize);
    printf("  Max Threads / Block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Max Grid Dimensions  : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max Block Dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("=========================================================\n\n");
}
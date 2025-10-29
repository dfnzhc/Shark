/**
 * @File Test.cu
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/29
 * @Brief This file is part of Shark.
 */

#include <iostream>

// System includes
#include <cassert>
#include <cstdio>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#include "Shark/Shark.hpp"
using namespace SKT;

__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d -> '%d'\n",
           blockIdx.y * gridDim.x + blockIdx.x,
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x,
           val, TestFunc2());
}

int main(int argc, char **argv)
{
    int            devID;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // Get GPU information
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute capability %d.%d\n", devID, props.name, props.major, props.minor);

    printf("什么？printf() is called. Output:\n\n");

    // Kernel configuration, where a two-dimensional grid and
    // three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel<<<dimGrid, dimBlock>>>(10);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
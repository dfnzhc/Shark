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

extern bool TestCommonMath();
extern bool TestVectorMath();

int main(int argc, char** argv)
{
    std::cout << "****************************\n";
    
    std::cout << "Common Math 测试: " << (TestCommonMath() ? "成功\n" : "失败.\n");
    
    std::cout << "----------------------------\n";
    
    std::cout << "Vector Math 测试: " << (TestVectorMath() ? "成功\n" : "失败.\n");

    std::cout << "****************************\n";

    return EXIT_SUCCESS;
}

/**
 * @File Defines.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/27
 * @Brief This file is part of Shark.
 */

#pragma once

#include <cmath>
#include <climits>
#include <cfloat>
#include <limits>
#include <cassert>

#define SKT_VERSION_MAJOR 0
#define SKT_VERSION_MINOR 0
#define SKT_VERSION_PATCH 1

// -------------------------
// 平台检测
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
#  define SKT_PLATFORM_WINDOWS 1
#elif defined(__unix__) || defined(__unix) || defined(__linux__)
#  define SKT_PLATFORM_LINUX 1
#elif defined(__APPLE__) || defined(__MACH__)
#  define SKT_PLATFORM_MACOS 1
#else
#  error "不支持的平台. Shark 目前支持 Windows, Linux, and macOS."
#endif

// -------------------------
// 编译器检测
#if defined(_MSC_VER) && defined(__clang__)
#  define SKT_COMPILER_CLANG_CL 1  // MSVC Clang (clang-cl)
#elif defined(_MSC_VER) && !defined(__clang__)
#  define SKT_COMPILER_MSVC 1      // MSVC (cl)
#elif defined(__clang__)
#  define SKT_COMPILER_CLANG 1     // Standard Clang
#elif defined(__GNUC__) && !defined(__clang__)
#  define SKT_COMPILER_GCC 1       // GCC
#elif defined(__NVCC__) || defined(__CUDACC__)
#  define SKT_COMPILER_NVCC 1      // NVCC
#else
#  error "不支持的编译器. Shark 目前支持 MSVC, Clang, GCC, and NVCC."
#endif

// -------------------------
// C++版本检测
#if defined(SKT_COMPILER_MSVC) || defined(SKT_COMPILER_CLANG_CL)
#  define SKT_CPLUSPLUS _MSVC_LANG
#else
#  define SKT_CPLUSPLUS __cplusplus
#endif

static_assert(SKT_CPLUSPLUS >= 202002L, "Shark 必须使用 C++20 或更新的标准.");

// -------------------------
// forceinline
#if defined(SKT_COMPILER_MSVC) || defined(SKT_COMPILER_CLANG_CL)
#  define SKT_FORCE_INLINE __forceinline
#elif defined(SKT_COMPILER_GCC) || defined(SKT_COMPILER_CLANG)
#  define SKT_FORCE_INLINE inline __attribute__((__always_inline__))
#elif defined(SKT_COMPILER_NVCC)
#  define SKT_FORCE_INLINE __forceinline__
#else
#  define SKT_FORCE_INLINE inline
#endif

// -------------------------
// noinline
#if defined(SKT_COMPILER_MSVC) || defined(SKT_COMPILER_CLANG_CL)
#  define SKT_NOINLINE __declspec(noinline)
#elif defined(SKT_COMPILER_GCC) || defined(SKT_COMPILER_CLANG)
#  define SKT_NOINLINE __attribute__((__noinline__))
#elif defined(SKT_COMPILER_NVCC)
#  define SKT_NOINLINE __noinline__
#else
#  define SKT_NOINLINE
#endif

// -------------------------
// GPU 代码设置
#if defined(__CUDA_ARCH__)
#define SKT_GPU_CODE 1
#else
#define SKT_CPU_CODE 1
#endif

#if defined(SKT_COMPILER_NVCC)
#define SKT_CPU_GPU __host__ __device__
#define SKT_GPU __device__
#else
#  define SKT_CPU_GPU
#  define SKT_GPU
#endif

#define SKT_FUNC_DECL SKT_CPU_GPU constexpr
#define SKT_FUNC SKT_CPU_GPU inline constexpr

// -------------------------
// likely & unlikely
#if defined(SKT_COMPILER_GCC) || defined(SKT_COMPILER_CLANG)
#  define SKT_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define SKT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define SKT_LIKELY(x)   (x) [[likely]]
#  define SKT_UNLIKELY(x) (x) [[unlikely]]
#endif

// -------------------------
// has builtin
#ifdef __has_builtin
#  define SKT_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define SKT_HAS_BUILTIN(x) 0
#endif

// -------------------------
// Assume
#if defined(SKT_COMPILER_MSVC) || defined(SKT_COMPILER_CLANG_CL)
#  define SKT_ASSUME(condition) __assume(condition)
#elif defined(SKT_COMPILER_CLANG)
#  if SKT_HAS_BUILTIN(__builtin_assume)
#    define SKT_ASSUME(condition) __builtin_assume(condition)
#  else
#    define SKT_ASSUME(condition)        \
        do {                             \
            if (!(condition))            \
                __builtin_unreachable(); \
        } while (0)
#  endif
#elif defined(SKT_COMPILER_GCC)
#  define SKT_ASSUME(condition)            \
      do {                                 \
          if (SKT_UNLIKELY(!(condition)))  \
              __builtin_unreachable();     \
      } while (0)
#else
#  define SKT_ASSUME(condition) ((void)0)
#endif

// -------------------------
// Alignas
#if defined(SKT_COMPILER_MSVC) || defined(SKT_COMPILER_CLANG_CL)
#  define SKT_ALIGNAS(N) __declspec(align(N))
#elif defined(SKT_COMPILER_GCC) || defined(SKT_COMPILER_CLANG)
#  define SKT_ALIGNAS(N) __attribute__((aligned(N)))
#elif defined(SKT_COMPILER_NVCC)
#  define SKT_ALIGNAS(N) __align__(N)
#else
#  define SKT_ALIGNAS(N) alignas(N)
#endif

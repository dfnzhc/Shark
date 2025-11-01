/**
 * @File Check.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/1
 * @Brief This file is part of Shark.
 */

#pragma once

#include "Defines.hpp"
#include <cstdio>

namespace Bee
{

    #ifdef SKT_GPU_CODE

    #ifdef SKT_CHECK
    #undef SKT_CHECK
    #endif
    #ifdef SKT_CHECK_OP
    #undef SKT_CHECK_OP
    #endif

    #define SKT_CHECK(x) assert(x)
    #define SKT_CHECK_OP(a, b, op) assert((a)op(b))

    #else
    #ifndef SKT_CHECK
    #define SKT_CHECK(x) assert(x)
    #endif
    
    #ifndef SKT_CHECK_OP
    #define CHECK_IMPL(a, b, op)                                                                                    \
    do {                                                                                                            \
        auto va = (a);                                                                                              \
        auto vb = (b);                                                                                              \
        if (!(va op vb)) printf("[SKT] Check failed: %s " #op " %s with %s = %s, %s = %s", #a, #b, #a, va, #b, vb); \
    } while (false)
    #endif
    #endif  // SKT_GPU_CODE

    #define SKT_CHECK_EQ(a, b) SKT_CHECK_OP(a, b, ==)
    #define SKT_CHECK_NE(a, b) SKT_CHECK_OP(a, b, !=)
    #define SKT_CHECK_GT(a, b) SKT_CHECK_OP(a, b, >)
    #define SKT_CHECK_GE(a, b) SKT_CHECK_OP(a, b, >=)
    #define SKT_CHECK_LT(a, b) SKT_CHECK_OP(a, b, <)
    #define SKT_CHECK_LE(a, b) SKT_CHECK_OP(a, b, <=)

    #if SKT_DEBUG_BUILD
    
    #define SKT_DCHECK(x) (SKT_CHECK(x))
    #define SKT_DCHECK_EQ(a, b) SKT_CHECK_EQ(a, b)
    #define SKT_DCHECK_NE(a, b) SKT_CHECK_NE(a, b)
    #define SKT_DCHECK_GT(a, b) SKT_CHECK_GT(a, b)
    #define SKT_DCHECK_GE(a, b) SKT_CHECK_GE(a, b)
    #define SKT_DCHECK_LT(a, b) SKT_CHECK_LT(a, b)
    #define SKT_DCHECK_LE(a, b) SKT_CHECK_LE(a, b)
    
    #else
    
    #define EMPTY_CHECK do { } while (false)

    #define SKT_DCHECK(x) EMPTY_CHECK
    #define SKT_DCHECK_EQ(a, b) EMPTY_CHECK
    #define SKT_DCHECK_NE(a, b) EMPTY_CHECK
    #define SKT_DCHECK_GT(a, b) EMPTY_CHECK
    #define SKT_DCHECK_GE(a, b) EMPTY_CHECK
    #define SKT_DCHECK_LT(a, b) EMPTY_CHECK
    #define SKT_DCHECK_LE(a, b) EMPTY_CHECK
    
    #endif

} // namespace Bee

/**
 * @File Common.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/27
 * @Brief This file is part of Shark.
 */

#pragma once

#include "Shark/Core/Concepts.hpp"

#include "Constants.hpp"
#include <algorithm>

namespace SKT
{
    // -------------------- 
    // 基础函数

    template <ArithType T>
    SKT_FUNC T Abs(T x)
    {
        return std::abs(x);
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto Min(A x, B y)
    {
        using R = CommonType<A, B>;
        return std::min(As<R>(x), As<R>(y));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto Max(A x, B y)
    {
        using R = CommonType<A, B>;
        return std::max(As<R>(x), As<R>(y));
    }

    template <ArithType T, ArithType U, ArithType... Ts>
        requires ConvertibleTo<U, T> && AllConvertibleTo<T, Ts...>
    constexpr auto Min(T a, U b, Ts... vals) noexcept
    {
        using R = CommonType<T, U>;

        const auto m = As<R>(a < b ? a : b);
        if constexpr (sizeof...(vals) > 0)
        {
            return Min(m, As<R>(vals)...);
        }
        return m;
    }

    template <ArithType T, ArithType U, ArithType... Ts>
        requires ConvertibleTo<U, T> && AllConvertibleTo<T, Ts...>
    constexpr auto Max(T a, U b, Ts... vals) noexcept
    {
        using R = CommonType<T, U>;

        const auto m = As<R>(a > b ? a : b);
        if constexpr (sizeof...(vals) > 0)
        {
            return Max(m, As<R>(vals)...);
        }
        return m;
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto FMod(A x, B y)
    {
        using F = CommonFloatType<A, B>;

        return std::fmod(As<F>(x), As<F>(y));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto Remainder(A x, B y)
    {
        using F = CommonFloatType<A, B>;

        return std::remainder(As<F>(x), As<F>(y));
    }

    template <ArithType A, ArithType B, ArithType C>
    SKT_FUNC auto FMA(A x, B y, C z)
    {
        using F = CommonFloatType<A, B, C>;

        return std::fma(As<F>(x), As<F>(y), As<F>(z));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto FDim(A x, B y)
    {
        using F = CommonFloatType<A, B>;

        return std::fdim(As<F>(x), As<F>(y));
    }

    // -------------------- 
    // 三角函数

    template <ArithType T>
    SKT_FUNC auto Sin(T x)
    {
        using F = MapFloatType<T>;

        return std::sin(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Cos(T x)
    {
        using F = MapFloatType<T>;

        return std::cos(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Tan(T x)
    {
        using F = MapFloatType<T>;

        return std::tan(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcSin(T x)
    {
        using F = MapFloatType<T>;

        return std::asin(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcCos(T x)
    {
        using F = MapFloatType<T>;

        return std::acos(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcTan(T x)
    {
        using F = MapFloatType<T>;

        return std::atan(As<F>(x));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto ArcTan2(A x, B y)
    {
        using F = CommonFloatType<A, B>;

        return std::atan2(As<F>(x), As<F>(y));
    }

    template <ArithType T>
    SKT_FUNC auto Sinh(T x)
    {
        using F = MapFloatType<T>;

        return std::sinh(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Cosh(T x)
    {
        using F = MapFloatType<T>;

        return std::cosh(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Tanh(T x)
    {
        using F = MapFloatType<T>;

        return std::tanh(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcSinh(T x)
    {
        using F = MapFloatType<T>;

        return std::asinh(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcCosh(T x)
    {
        using F = MapFloatType<T>;

        return std::acosh(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ArcTanh(T x)
    {
        using F = MapFloatType<T>;

        return std::atanh(As<F>(x));
    }

    // -------------------- 
    // 指数函数

    template <ArithType T>
    SKT_FUNC auto Exp(T x)
    {
        using F = MapFloatType<T>;

        return std::exp(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Exp2(T x)
    {
        using F = MapFloatType<T>;

        return std::exp2(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto ExpM1(T x)
    {
        using F = MapFloatType<T>;

        return std::expm1(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Log(T x)
    {
        using F = MapFloatType<T>;

        return std::log(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Log10(T x)
    {
        using F = MapFloatType<T>;

        return std::log10(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Log2(T x)
    {
        using F = MapFloatType<T>;

        return std::log2(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Log1p(T x)
    {
        using F = MapFloatType<T>;

        return std::log1p(As<F>(x));
    }

    // -------------------- 
    // 幂函数

    template <ArithType A, ArithType B>
    SKT_FUNC auto Pow(A x, B y)
    {
        using F = CommonFloatType<A, B>;

        return std::pow(As<F>(x), As<F>(y));
    }

    template <ArithType T>
    SKT_FUNC auto Sqrt(T x)
    {
        using F = MapFloatType<T>;

        return std::sqrt(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Cbrt(T x)
    {
        using F = MapFloatType<T>;

        return std::cbrt(As<F>(x));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto Hypot(A x, B y)
    {
        using F = CommonFloatType<A, B>;
        return std::hypot(As<F>(x), As<F>(y));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto InvHypot(A x, B y)
    {
        using F = CommonFloatType<A, B>;
        
        #if SKT_GPU_CODE
        return rhypot(As<F>(x), As<F>(y));
        #else
        return As<F>(1) / std::hypot(As<F>(x), As<F>(y));
        #endif
    }

    // -------------------- 
    // 误差与 gamma 函数

    template <ArithType T>
    SKT_FUNC auto Erf(T x)
    {
        using F = MapFloatType<T>;

        return std::erf(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Erfc(T x)
    {
        using F = MapFloatType<T>;

        return std::erfc(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto tGmma(T x)
    {
        using F = MapFloatType<T>;

        return std::tgamma(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto lGmma(T x)
    {
        using F = MapFloatType<T>;

        return std::lgamma(As<F>(x));
    }

    // -------------------- 
    // 浮点数操作

    template <ArithType T>
    SKT_FUNC auto Ceil(T x)
    {
        using F = MapFloatType<T>;

        return std::ceil(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Floor(T x)
    {
        using F = MapFloatType<T>;

        return std::floor(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Trunc(T x)
    {
        using F = MapFloatType<T>;

        return std::trunc(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Round(T x)
    {
        using F = MapFloatType<T>;

        return std::round(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto NearbyInt(T x)
    {
        using F = MapFloatType<T>;

        return std::nearbyint(As<F>(x));
    }

    template <ArithType T>
    SKT_FUNC auto Frexp(T x, int* exp)
    {
        using F = MapFloatType<T>;

        return std::frexp(As<F>(x), exp);
    }

    template <ArithType T>
    SKT_FUNC auto Ldexp(T x, int exp)
    {
        using F = MapFloatType<T>;

        return std::ldexp(As<F>(x), exp);
    }

    template <FloatType F>
    SKT_FUNC auto Modf(F x, F* iptr)
    {
        return std::modf(x, iptr);
    }

    template <ArithType T>
    SKT_FUNC int Logb(T x)
    {
        using F = MapFloatType<T>;

        return std::ilogb(As<F>(x));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto NextAfter(A form, B to)
    {
        using F = CommonFloatType<A, B>;

        return std::nextafter(As<F>(form), As<F>(to));
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto CopySign(A mag, B sgn)
    {
        using F = CommonFloatType<A, B>;

        return std::copysign(As<F>(mag), As<F>(sgn));
    }
} // namespace SKT

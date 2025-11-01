/**
 * @File Common.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/27
 * @Brief This file is part of Shark.
 */

#pragma once

#include "Shark/Core/Concepts.hpp"
#include "Shark/Core/Check.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <numeric>

#ifdef SKT_GPU_CODE
#include <cuda_runtime_api.h>
#endif

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

    template <IntegralType T>
    SKT_FUNC T Mod(T a, T b) noexcept
    {
        T result = a - (a / b) * b;
        return As<T>((result < 0) ? result + b : result);
    }

    template <FloatType A, FloatType B>
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

        #ifdef SKT_GPU_CODE
        return ::rhypot(As<F>(x), As<F>(y));
        #else
        return As<F>(1) / Hypot(As<F>(x), As<F>(y));
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

    template <typename T>
    SKT_FUNC auto IsFinite(T)
    {
        return false;
    }

    SKT_FUNC auto IsFinite(FloatType auto x)
    {
        #ifdef SKT_GPU_CODE
        return isnan(x);
        #else
        return std::isfinite(x);
        #endif
    }

    template <typename T>
    SKT_FUNC auto IsInf(T)
    {
        return false;
    }

    SKT_FUNC auto IsInf(FloatType auto x)
    {
        #ifdef SKT_GPU_CODE
        return isinf(x);
        #else
        return std::isinf(x);
        #endif
    }

    template <typename T>
    SKT_FUNC auto IsNaN(T)
    {
        return false;
    }

    SKT_FUNC auto IsNaN(FloatType auto x)
    {
        #ifdef SKT_GPU_CODE
        return isnan(x);
        #else
        return std::isnan(x);
        #endif
    }

    template <FloatType T>
    SKT_FUNC auto FloatToBits(T x) -> std::conditional_t<sizeof(T) == 4u, u32, u64>
    {
        if constexpr (sizeof(T) == 4)
        {
            #ifdef SKT_GPU_CODE
            return __float_as_uint(x);
            #else
            return BitCast<u32>(x);
            #endif
        }
        else
        {
            #ifdef SKT_GPU_CODE
            return __double_as_longlong(x);
            #else
            return BitCast<u64>(x);
            #endif
        }
    }

    template <UnsignedType T>
    SKT_FUNC auto BitsToFloat(T x) -> std::conditional_t<sizeof(T) == 4u, f32, f64>
    {
        if constexpr (sizeof(T) == 4)
        {
            #ifdef SKT_GPU_CODE
            return __uint_as_float(x);
            #else
            return BitCast<f32>(x);
            #endif
        }
        else
        {
            #ifdef SKT_GPU_CODE
            return __longlong_as_double(x);
            #else
            return BitCast<f64>(x);
            #endif
        }
    }

    template <FloatType T>
    SKT_FUNC int Exponent(T x) noexcept
    {
        if constexpr (sizeof(T) == 4)
        {
            return ((FloatToBits(x) >> 23) & 0xFF) - 127;
        }
        else
        {
            return ((FloatToBits(x) >> 52) & 0x7FF) - 1023;
        }
    }

    template <FloatType T>
    SKT_FUNC auto Significand(T x) noexcept -> std::conditional_t<sizeof(T) == 4u, u32, u64>
    {
        if constexpr (sizeof(T) == 4)
        {
            return FloatToBits(x) & ((1 << 23) - 1);
        }
        else
        {
            return FloatToBits(x) & ((1ull << 52) - 1);
        }
    }

    template <FloatType T>
    SKT_FUNC auto SignBit(T x) noexcept -> std::conditional_t<sizeof(T) == 4u, u32, u64>
    {
        if constexpr (sizeof(T) == 4)
        {
            return FloatToBits(x) & 0x80000000;
        }
        else
        {
            return FloatToBits(x) & 0x8000000000000000;
        }
    }

    template <FloatType T>
    SKT_FUNC T NextFloatUp(T x) noexcept
    {
        if (IsInf(x) && x > 0.f)
            return x;

        if (x == -0.f)
            x = 0.f;

        auto ui = FloatToBits(x);
        if (x >= 0)
            ++ui;
        else
            --ui;
        return BitsToFloat(ui);
    }

    template <FloatType T>
    SKT_FUNC T NextFloatDown(T x) noexcept
    {
        if (IsInf(x) && x < 0.)
            return x;

        if (x == 0.f)
            x = -0.f;

        auto ui = FloatToBits(x);
        if (x > 0)
            --ui;
        else
            ++ui;
        return BitsToFloat(ui);
    }

    template <FloatType T>
    SKT_FUNC T AddRoundUp(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fadd_ru(a, b);
        }
        else
        {
            return __dadd_ru(a, b);
        }
        #else
        return NextFloatUp(a + b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T AddRoundDown(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fadd_rd(a, b);
        }
        else
        {
            return __dadd_rd(a, b);
        }
        #else
        return NextFloatDown(a + b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T SubRoundUp(T a, T b)
    {
        return AddRoundUp(a, -b);
    }

    template <FloatType T>
    SKT_FUNC T SubRoundDown(T a, T b)
    {
        return AddRoundDown(a, -b);
    }

    template <FloatType T>
    SKT_FUNC T MulRoundUp(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fmul_ru(a, b);
        }
        else
        {
            return __dmul_ru(a, b);
        }
        #else
        return NextFloatUp(a * b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T MulRoundDown(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fmul_rd(a, b);
        }
        else
        {
            return __dmul_rd(a, b);
        }
        #else
        return NextFloatDown(a * b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T DivRoundUp(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fdiv_ru(a, b);
        }
        else
        {
            return __ddiv_ru(a, b);
        }
        #else
        return NextFloatUp(a / b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T DivRoundDown(T a, T b)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fdiv_rd(a, b);
        }
        else
        {
            return __ddiv_rd(a, b);
        }
        #else
        return NextFloatDown(a / b);
        #endif
    }

    template <FloatType T>
    SKT_FUNC T SqrtRoundUp(T x)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fsqrt_ru(x);
        }
        else
        {
            return __dsqrt_ru(x);
        }
        #else
        return NextFloatUp(Sqrt(x));
        #endif
    }

    template <FloatType T>
    SKT_FUNC T SqrtRoundDown(T x)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fsqrt_rd(x);
        }
        else
        {
            return __dsqrt_rd(x);
        }
        #else
        return NextFloatDown(Sqrt(x));
        #endif
    }

    template <FloatType T>
    SKT_FUNC T FMARoundUp(T a, T b, T c)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fma_ru(a, b, c);
        }
        else
        {
            return __fma_ru(a, b, c);
        }
        #else
        return NextFloatUp(FMA(a, b, c));
        #endif
    }

    template <FloatType T>
    SKT_FUNC T FMARoundDown(T a, T b, T c)
    {
        #ifdef SKT_GPU_CODE
        if constexpr (sizeof(T) == 4)
        {
            return __fma_ru(a, b, c);
        }
        else
        {
            return __fma_ru(a, b, c);
        }
        #else
        return NextFloatDown(FMA(a, b, c));
        #endif
    }

    SKT_FUNC f64 Gamma(int n) noexcept
    {
        return (n * kEpsilonD) / (1 - n * kEpsilonD);
    }


    template <ArithType A, ArithType B, ArithType C>
    SKT_FUNC A Clamp(A x, B lo, C hi)
    {
        return std::clamp(x, As<A>(lo), As<A>(hi));
    }

    template <ArithType A>
    SKT_FUNC auto Clamp01(A x)
    {
        return std::clamp(x, As<A>(0), As<A>(1));
    }

    // -------------------- 
    // 其他方法

    template <FloatType T>
    SKT_FUNC auto AlmostZero(T x) noexcept
    {
        return T(1) - x * x == T(1);
    }

    template <ArithType T>
    SKT_FUNC auto Sinc(T x)
    {
        using F = MapFloatType<T>;

        auto fx = As<F>(x);
        if (AlmostZero(fx))
            return F(1);

        return Sin(fx) / fx;
    }

    template <ArithType T>
    SKT_FUNC auto Sqr(T x)
    {
        return x * x;
    }

    template <ArithType T>
    SKT_FUNC auto Degree2Radian(T degree) noexcept
    {
        using F = MapFloatType<T>;
        return As<F>(kPi / 180) * As<F>(degree);
    }

    template <ArithType T>
    SKT_FUNC auto Radian2Degree(T radian) noexcept
    {
        using F = MapFloatType<T>;
        return As<F>(kInvPi * 180) * As<F>(radian);
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto GCD(A x, B y)
    {
        return std::gcd(x, y);
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto LCM(A x, B y)
    {
        return std::lcm(x, y);
    }

    SKT_FUNC auto GCD(auto x, auto... xs)
    {
        return ((x = GCD(x, xs)), ...);
    }

    SKT_FUNC auto LCM(auto x, auto... xs)
    {
        return ((x = LCM(x, xs)), ...);
    }

    template <ArithType A, ArithType B>
    SKT_FUNC auto Midpoint(A a, B b) noexcept
    {
        using F = CommonFloatType<A, B>;

        auto fa = As<F>(a);
        auto fb = As<F>(b);
        return std::midpoint(As<F>(a), As<F>(b));
    }

    template <ArithType A, ArithType B, FloatType C>
    SKT_FUNC auto Lerp(A a, B b, C x)
    {
        using F = CommonFloatType<A, B, C>;

        auto fa = As<F>(a);
        auto fb = As<F>(b);
        auto fx = As<F>(x);
        return (As<F>(1) - fx) * fa + fx * fb;
    }

    template <ArithType T, ArithType C>
    SKT_FUNC T EvaluatePolynomial(T, C c) noexcept
    {
        return As<T>(c);
    }

    template <ArithType T, ArithType C, ArithType... Args>
    SKT_FUNC T EvaluatePolynomial(T t, C c, Args... cRemaining)
    {
        return FMA(t, As<T>(EvaluatePolynomial(t, cRemaining...)), As<T>(c));
    }

    template <FloatType T>
    SKT_FUNC bool Quadratic(T a, T b, T c, T* t0, T* t1)
    {
        if (a == 0)
        {
            if (b == 0)
                return false;

            *t0 = *t1 = -c / b;
            return true;
        }

        auto discriminant = b * b - As<T>(4) * a * c;
        if (discriminant < 0)
            return false;
        auto rootDiscriminant = Sqrt(discriminant);

        auto q = -As<T>(0.5) * (b + CopySign(rootDiscriminant, b));
        *t0    = q / a;
        *t1    = c / q;

        if (*t0 > *t1)
            std::swap(*t0, *t1);

        return true;
    }


    SKT_FUNC f32 FastExp(f32 x)
    {
        #ifdef SKT_GPU_CODE
        return __expf(x);
        #else
        f32 xp  = x * 1.442695041f;
        f32 fxp = Floor(xp), f = xp - fxp;
        int i   = As<int>(fxp);

        f32 twoToF   = EvaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f, 0.0781455737f);
        int exponent = Exponent(twoToF) + i;
        if (exponent < -126)
            return 0;
        if (exponent > 127)
            return kInfinityF;

        u32 bits = FloatToBits(twoToF);
        bits &= 0b10000000011111111111111111111111u;
        bits |= (exponent + 127) << 23;
        return BitsToFloat(bits);
        #endif
    }

    SKT_FUNC f32 FastSqrt(f32 x0)
    {
        SKT_CHECK(x0 >= 0);

        union
        {
            i32 ix;
            f32 x;
        } u{};

        u.x  = x0;
        u.ix = 0x1fbb3f80 + (u.ix >> 1);
        u.x  = 0.5f * (u.x + x0 / u.x);
        u.x  = 0.5f * (u.x + x0 / u.x);
        return u.x;
    }

    SKT_FUNC f32 FastCbrt(f32 x0)
    {
        SKT_CHECK(x0 >= 0);

        union
        {
            i32 ix;
            f32 x;
        } u{};

        u.x  = x0;
        u.ix = u.ix / 4 + u.ix / 16;
        u.ix = u.ix + u.ix / 16;
        u.ix = u.ix + u.ix / 256;
        u.ix = 0x2a5137a0 + u.ix;
        u.x  = 0.33333333f * (2.0f * u.x + x0 / (u.x * u.x));
        u.x  = 0.33333333f * (2.0f * u.x + x0 / (u.x * u.x));
        return u.x;
    }

    template <FloatType T>
    SKT_FUNC T FastInvSqrt(T x0)
    {
        SKT_CHECK(x0 > 0);

        if constexpr (sizeof(T) == 4)
        {
            union
            {
                i32 ix;
                f32 x;
            } u{};

            u.x       = x0;
            f32 xHalf = 0.5f * u.x;
            u.ix      = 0x5f37599e - (u.ix >> 1);
            u.x       = u.x * (1.5f - xHalf * u.x * u.x);
            u.x       = u.x * (1.5f - xHalf * u.x * u.x);
            return u.x;
        }
        else
        {
            union
            {
                i64 ix;
                f64 x;
            } u{};

            u.x       = x0;
            f64 xHalf = 0.5 * u.x;
            u.ix      = 0x5fe6ec85e8000000LL - (u.ix >> 1);
            u.x       = u.x * (1.5 - xHalf * u.x * u.x);
            u.x       = u.x * (1.5 - xHalf * u.x * u.x);
            return u.x;
        }
    }

    template <FloatType T>
    SKT_FUNC T Gaussian(T x, T mu = 0, T sigma = 1)
    {
        return As<T>(1) / Sqrt(TwoPi<T>() * sigma * sigma) * FastExp(-Sqr(x - mu) / (As<T>(2) * sigma * sigma));
    }

    template <FloatType T>
    SKT_FUNC T GaussianIntegral(T x0, T x1, T mu = 0, T sigma = 1)
    {
        T sigmaRoot2 = sigma * Sqrt2<T>();
        return As<T>(0.5) * (Erf((mu - x0) / sigmaRoot2) - Erf((mu - x1) / sigmaRoot2));
    }


} // namespace SKT

/**
 * @File CommonTest.cpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/1
 * @Brief This file is part of Shark.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include <Shark/Shark.hpp>

using namespace SKT;

#define EXPECT_NEAR_DOUBLE(val1, val2) EXPECT_NEAR(val1, val2, kEpsilonD)
#define EXPECT_NEAR_FLOAT(val1, val2) EXPECT_NEAR(val1, val2, kEpsilonF)

TEST(BasicFunctionsTest, AbsFunction)
{
    EXPECT_EQ(Abs(-5), 5);
    EXPECT_EQ(Abs(5), 5);
    EXPECT_EQ(Abs(0), 0);

    EXPECT_NEAR_DOUBLE(Abs(-3.14), 3.14);
    EXPECT_NEAR_DOUBLE(Abs(3.14), 3.14);
    EXPECT_NEAR_DOUBLE(Abs(0.0), 0.0);

    EXPECT_EQ(Abs(std::numeric_limits<int>::min() + 1), std::numeric_limits<int>::max());
}

TEST(BasicFunctionsTest, MinFunction)
{
    EXPECT_EQ(Min(3, 5), 3);
    EXPECT_EQ(Min(5, 3), 3);
    EXPECT_EQ(Min(3, 3), 3);

    EXPECT_NEAR_DOUBLE(Min(3.14, 2.71), 2.71);
    EXPECT_NEAR_DOUBLE(Min(-1.5, -2.5), -2.5);

    EXPECT_NEAR_DOUBLE(Min(3, 2.5), 2.5);
    EXPECT_NEAR_DOUBLE(Min(2.5, 3), 2.5);

    EXPECT_EQ(Min(5, 3, 8, 1, 9), 1);
    EXPECT_NEAR_DOUBLE(Min(3.14, 2.71, 1.41, 0.57), 0.57);
}

TEST(BasicFunctionsTest, MaxFunction)
{
    EXPECT_EQ(Max(3, 5), 5);
    EXPECT_EQ(Max(5, 3), 5);
    EXPECT_EQ(Max(3, 3), 3);

    EXPECT_NEAR_DOUBLE(Max(3.14, 2.71), 3.14);
    EXPECT_NEAR_DOUBLE(Max(-1.5, -2.5), -1.5);

    EXPECT_NEAR_DOUBLE(Max(3, 2.5), 3.0);
    EXPECT_NEAR_DOUBLE(Max(2.5, 3), 3.0);

    EXPECT_EQ(Max(5, 3, 8, 1, 9), 9);
    EXPECT_NEAR_DOUBLE(Max(3.14, 2.71, 1.41, 0.57), 3.14);
}

TEST(BasicFunctionsTest, FModFunction)
{
    EXPECT_NEAR_DOUBLE(FMod(5.3, 2.0), std::fmod(5.3, 2.0));
    EXPECT_NEAR_DOUBLE(FMod(-5.3, 2.0), std::fmod(-5.3, 2.0));
    EXPECT_NEAR_DOUBLE(FMod(5.3, -2.0), std::fmod(5.3, -2.0));

    EXPECT_NEAR_DOUBLE(FMod(7, 3.0), std::fmod(7.0, 3.0));
}

TEST(BasicFunctionsTest, RemainderFunction)
{
    EXPECT_NEAR_DOUBLE(Remainder(5.3, 2.0), std::remainder(5.3, 2.0));
    EXPECT_NEAR_DOUBLE(Remainder(-5.3, 2.0), std::remainder(-5.3, 2.0));
    EXPECT_NEAR_DOUBLE(Remainder(5.3, -2.0), std::remainder(5.3, -2.0));
}

TEST(BasicFunctionsTest, FMAFunction)
{
    double x = 2.0, y = 3.0, z = 1.0;
    EXPECT_NEAR_DOUBLE(FMA(x, y, z), std::fma(x, y, z));

    double a = 1e16, b = 1.0, c = -1e16;
    EXPECT_NEAR_DOUBLE(FMA(a, b, c), std::fma(a, b, c));
}

TEST(BasicFunctionsTest, FDimFunction)
{
    EXPECT_NEAR_DOUBLE(FDim(5.0, 3.0), std::fdim(5.0, 3.0));
    EXPECT_NEAR_DOUBLE(FDim(3.0, 5.0), std::fdim(3.0, 5.0));
    EXPECT_NEAR_DOUBLE(FDim(3.0, 3.0), std::fdim(3.0, 3.0));
}

TEST(TrigonometricFunctionsTest, BasicTrigFunctions)
{
    double angles[] = {0.0, kPi / 6, kPi / 4, kPi / 3, kPi / 2, kPi, 2 * kPi};

    for (double angle : angles)
    {
        EXPECT_NEAR_DOUBLE(Sin(angle), std::sin(angle));
        EXPECT_NEAR_DOUBLE(Cos(angle), std::cos(angle));
        if (std::abs(std::cos(angle)) > kEpsilonF)
        {
            EXPECT_NEAR_DOUBLE(Tan(angle), std::tan(angle));
        }
    }
}

TEST(TrigonometricFunctionsTest, InverseTrigFunctions)
{
    double values[] = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(ArcSin(val), std::asin(val));
        EXPECT_NEAR_DOUBLE(ArcCos(val), std::acos(val));
    }

    double tan_values[] = {-10.0, -1.0, 0.0, 1.0, 10.0};
    for (double val : tan_values)
    {
        EXPECT_NEAR_DOUBLE(ArcTan(val), std::atan(val));
    }
}

TEST(TrigonometricFunctionsTest, ArcTan2Function)
{
    double test_cases[][2] = {
            {1.0, 1.0},
            {1.0, -1.0},
            {-1.0, 1.0},
            {-1.0, -1.0},
            {0.0, 1.0},
            {0.0, -1.0},
            {1.0, 0.0},
            {-1.0, 0.0}
    };

    for (auto& test_case : test_cases)
    {
        EXPECT_NEAR_DOUBLE(ArcTan2(test_case[0], test_case[1]), std::atan2(test_case[0], test_case[1]));
    }
}

TEST(TrigonometricFunctionsTest, HyperbolicFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Sinh(val), std::sinh(val));
        EXPECT_NEAR_DOUBLE(Cosh(val), std::cosh(val));
        EXPECT_NEAR_DOUBLE(Tanh(val), std::tanh(val));
    }
}

TEST(TrigonometricFunctionsTest, InverseHyperbolicFunctions)
{
    double sinh_values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (double val : sinh_values)
    {
        EXPECT_NEAR_DOUBLE(ArcSinh(val), std::asinh(val));
    }

    double cosh_values[] = {1.0, 1.5, 2.0, 3.0};
    for (double val : cosh_values)
    {
        EXPECT_NEAR_DOUBLE(ArcCosh(val), std::acosh(val));
    }

    double tanh_values[] = {-0.9, -0.5, 0.0, 0.5, 0.9};
    for (double val : tanh_values)
    {
        EXPECT_NEAR_DOUBLE(ArcTanh(val), std::atanh(val));
    }
}

TEST(ExponentialFunctionsTest, ExpFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 10.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Exp(val), std::exp(val));
        EXPECT_NEAR_DOUBLE(Exp2(val), std::exp2(val));
        EXPECT_NEAR_DOUBLE(ExpM1(val), std::expm1(val));
    }
}

TEST(ExponentialFunctionsTest, LogFunctions)
{
    double values[] = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Log(val), std::log(val));
        EXPECT_NEAR_DOUBLE(Log10(val), std::log10(val));
        EXPECT_NEAR_DOUBLE(Log2(val), std::log2(val));
    }

    double log1p_values[] = {-0.9, -0.5, 0.0, 0.5, 1.0};
    for (double val : log1p_values)
    {
        EXPECT_NEAR_DOUBLE(Log1p(val), std::log1p(val));
    }
}

TEST(PowerFunctionsTest, PowFunction)
{
    double bases[]     = {0.5, 1.0, 2.0, 10.0};
    double exponents[] = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0};

    for (double base : bases)
    {
        for (double exp : exponents)
        {
            if (base > 0 || (base == 0 && exp > 0))
            {
                EXPECT_NEAR_DOUBLE(Pow(base, exp), std::pow(base, exp));
            }
        }
    }
}

TEST(PowerFunctionsTest, SqrtFunction)
{
    double values[] = {0.0, 0.25, 1.0, 4.0, 9.0, 16.0, 100.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Sqrt(val), std::sqrt(val));
    }
}

TEST(PowerFunctionsTest, CbrtFunction)
{
    double values[] = {-8.0, -1.0, 0.0, 1.0, 8.0, 27.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Cbrt(val), std::cbrt(val));
    }
}

TEST(PowerFunctionsTest, HypotFunction)
{
    double test_cases[][2] = {
            {3.0, 4.0},
            {5.0, 12.0},
            {8.0, 15.0},
            {0.0, 1.0},
            {1.0, 0.0}
    };

    for (auto& test_case : test_cases)
    {
        EXPECT_NEAR_DOUBLE(Hypot(test_case[0], test_case[1]), std::hypot(test_case[0], test_case[1]));
    }
}

TEST(ErrorGammaFunctionsTest, ErrorFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Erf(val), std::erf(val));
        EXPECT_NEAR_DOUBLE(Erfc(val), std::erfc(val));
    }
}

TEST(ErrorGammaFunctionsTest, GammaFunctions)
{
    double values[] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(tGmma(val), std::tgamma(val));
        EXPECT_NEAR_DOUBLE(lGmma(val), std::lgamma(val));
    }
}

TEST(FloatingPointTest, RoundingFunctions)
{
    double values[] = {-2.7, -2.3, -1.5, -0.5, 0.0, 0.5, 1.5, 2.3, 2.7};

    for (double val : values)
    {
        EXPECT_NEAR_DOUBLE(Ceil(val), std::ceil(val));
        EXPECT_NEAR_DOUBLE(Floor(val), std::floor(val));
        EXPECT_NEAR_DOUBLE(Trunc(val), std::trunc(val));
        EXPECT_NEAR_DOUBLE(Round(val), std::round(val));
        EXPECT_NEAR_DOUBLE(NearbyInt(val), std::nearbyint(val));
    }
}

TEST(FloatingPointTest, ManipulationFunctions)
{
    double val = 3.14159;
    int exp;

    double frac     = Frexp(val, &exp);
    double std_frac = std::frexp(val, &exp);
    EXPECT_NEAR_DOUBLE(frac, std_frac);

    EXPECT_NEAR_DOUBLE(Ldexp(0.5, 3), std::ldexp(0.5, 3));
    EXPECT_NEAR_DOUBLE(Scalbn(1.5, 4), std::scalbn(1.5, 4));

    EXPECT_EQ(Logb(8.0), std::ilogb(8.0));
}

TEST(FloatingPointTest, ModfFunction)
{
    double values[] = {3.14159, -2.71828, 0.0, 1.0, -1.0};

    for (double val : values)
    {
        double int_part, std_int_part;
        double frac_part     = Modf(val, &int_part);
        double std_frac_part = std::modf(val, &std_int_part);

        EXPECT_NEAR_DOUBLE(frac_part, std_frac_part);
        EXPECT_NEAR_DOUBLE(int_part, std_int_part);
    }
}

TEST(FloatingPointTest, NextAfterFunction)
{
    EXPECT_NEAR_DOUBLE(NextAfter(1.0, 2.0), std::nextafter(1.0, 2.0));
    EXPECT_NEAR_DOUBLE(NextAfter(1.0, 0.0), std::nextafter(1.0, 0.0));
}

TEST(FloatingPointTest, CopySignFunction)
{
    double test_cases[][2] = {
            {3.14, 1.0},
            {3.14, -1.0},
            {-3.14, 1.0},
            {-3.14, -1.0}
    };

    for (auto& test_case : test_cases)
    {
        EXPECT_NEAR_DOUBLE(CopySign(test_case[0], test_case[1]),
                           std::copysign(test_case[0], test_case[1]));
    }
}
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

TEST(BasicFunctionsTest, AbsFunction)
{
    EXPECT_EQ(Abs(-5), 5);
    EXPECT_EQ(Abs(5), 5);
    EXPECT_EQ(Abs(0), 0);

    EXPECT_DOUBLE_EQ(Abs(-3.14), 3.14);
    EXPECT_DOUBLE_EQ(Abs(3.14), 3.14);
    EXPECT_DOUBLE_EQ(Abs(0.0), 0.0);

    EXPECT_EQ(Abs(std::numeric_limits<int>::min() + 1), std::numeric_limits<int>::max());
}

TEST(BasicFunctionsTest, MinFunction)
{
    EXPECT_EQ(Min(3, 5), 3);
    EXPECT_EQ(Min(5, 3), 3);
    EXPECT_EQ(Min(3, 3), 3);

    EXPECT_DOUBLE_EQ(Min(3.14, 2.71), 2.71);
    EXPECT_DOUBLE_EQ(Min(-1.5, -2.5), -2.5);

    EXPECT_DOUBLE_EQ(Min(3, 2.5), 2.5);
    EXPECT_DOUBLE_EQ(Min(2.5, 3), 2.5);

    EXPECT_EQ(Min(5, 3, 8, 1, 9), 1);
    EXPECT_DOUBLE_EQ(Min(3.14, 2.71, 1.41, 0.57), 0.57);
}

TEST(BasicFunctionsTest, MaxFunction)
{
    EXPECT_EQ(Max(3, 5), 5);
    EXPECT_EQ(Max(5, 3), 5);
    EXPECT_EQ(Max(3, 3), 3);

    EXPECT_DOUBLE_EQ(Max(3.14, 2.71), 3.14);
    EXPECT_DOUBLE_EQ(Max(-1.5, -2.5), -1.5);

    EXPECT_DOUBLE_EQ(Max(3, 2.5), 3.0);
    EXPECT_DOUBLE_EQ(Max(2.5, 3), 3.0);

    EXPECT_EQ(Max(5, 3, 8, 1, 9), 9);
    EXPECT_DOUBLE_EQ(Max(3.14, 2.71, 1.41, 0.57), 3.14);
}

TEST(BasicFunctionsTest, FModFunction)
{
    EXPECT_DOUBLE_EQ(FMod(5.3, 2.0), std::fmod(5.3, 2.0));
    EXPECT_DOUBLE_EQ(FMod(-5.3, 2.0), std::fmod(-5.3, 2.0));
    EXPECT_DOUBLE_EQ(FMod(5.3, -2.0), std::fmod(5.3, -2.0));
}

TEST(BasicFunctionsTest, RemainderFunction)
{
    EXPECT_DOUBLE_EQ(Remainder(5.3, 2.0), std::remainder(5.3, 2.0));
    EXPECT_DOUBLE_EQ(Remainder(-5.3, 2.0), std::remainder(-5.3, 2.0));
    EXPECT_DOUBLE_EQ(Remainder(5.3, -2.0), std::remainder(5.3, -2.0));
}

TEST(BasicFunctionsTest, FMAFunction)
{
    double x = 2.0, y = 3.0, z = 1.0;
    EXPECT_DOUBLE_EQ(FMA(x, y, z), std::fma(x, y, z));

    double a = 1e16, b = 1.0, c = -1e16;
    EXPECT_DOUBLE_EQ(FMA(a, b, c), std::fma(a, b, c));
}

TEST(BasicFunctionsTest, FDimFunction)
{
    EXPECT_DOUBLE_EQ(FDim(5.0, 3.0), std::fdim(5.0, 3.0));
    EXPECT_DOUBLE_EQ(FDim(3.0, 5.0), std::fdim(3.0, 5.0));
    EXPECT_DOUBLE_EQ(FDim(3.0, 3.0), std::fdim(3.0, 3.0));
}

TEST(TrigonometricFunctionsTest, BasicTrigFunctions)
{
    double angles[] = {0.0, kPi / 6, kPi / 4, kPi / 3, kPi / 2, kPi, 2 * kPi};

    for (double angle : angles)
    {
        EXPECT_DOUBLE_EQ(Sin(angle), std::sin(angle));
        EXPECT_DOUBLE_EQ(Cos(angle), std::cos(angle));
        if (std::abs(std::cos(angle)) > kEpsilonF)
        {
            EXPECT_DOUBLE_EQ(Tan(angle), std::tan(angle));
        }
    }
}

TEST(TrigonometricFunctionsTest, InverseTrigFunctions)
{
    double values[] = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(ArcSin(val), std::asin(val));
        EXPECT_DOUBLE_EQ(ArcCos(val), std::acos(val));
    }

    double tan_values[] = {-10.0, -1.0, 0.0, 1.0, 10.0};
    for (double val : tan_values)
    {
        EXPECT_DOUBLE_EQ(ArcTan(val), std::atan(val));
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
        EXPECT_DOUBLE_EQ(ArcTan2(test_case[0], test_case[1]), std::atan2(test_case[0], test_case[1]));
    }
}

TEST(TrigonometricFunctionsTest, HyperbolicFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Sinh(val), std::sinh(val));
        EXPECT_DOUBLE_EQ(Cosh(val), std::cosh(val));
        EXPECT_DOUBLE_EQ(Tanh(val), std::tanh(val));
    }
}

TEST(TrigonometricFunctionsTest, InverseHyperbolicFunctions)
{
    double sinh_values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (double val : sinh_values)
    {
        EXPECT_DOUBLE_EQ(ArcSinh(val), std::asinh(val));
    }

    double cosh_values[] = {1.0, 1.5, 2.0, 3.0};
    for (double val : cosh_values)
    {
        EXPECT_DOUBLE_EQ(ArcCosh(val), std::acosh(val));
    }

    double tanh_values[] = {-0.9, -0.5, 0.0, 0.5, 0.9};
    for (double val : tanh_values)
    {
        EXPECT_DOUBLE_EQ(ArcTanh(val), std::atanh(val));
    }
}

TEST(ExponentialFunctionsTest, ExpFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 10.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Exp(val), std::exp(val));
        EXPECT_DOUBLE_EQ(Exp2(val), std::exp2(val));
        EXPECT_DOUBLE_EQ(ExpM1(val), std::expm1(val));
    }
}

TEST(ExponentialFunctionsTest, LogFunctions)
{
    double values[] = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Log(val), std::log(val));
        EXPECT_DOUBLE_EQ(Log10(val), std::log10(val));
        EXPECT_DOUBLE_EQ(Log2(val), std::log2(val));
    }

    double log1p_values[] = {-0.9, -0.5, 0.0, 0.5, 1.0};
    for (double val : log1p_values)
    {
        EXPECT_DOUBLE_EQ(Log1p(val), std::log1p(val));
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
                EXPECT_DOUBLE_EQ(Pow(base, exp), std::pow(base, exp));
            }
        }
    }
}

TEST(PowerFunctionsTest, SqrtFunction)
{
    double values[] = {0.0, 0.25, 1.0, 4.0, 9.0, 16.0, 100.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Sqrt(val), std::sqrt(val));
    }
}

TEST(PowerFunctionsTest, CbrtFunction)
{
    double values[] = {-8.0, -1.0, 0.0, 1.0, 8.0, 27.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Cbrt(val), std::cbrt(val));
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
        EXPECT_DOUBLE_EQ(Hypot(test_case[0], test_case[1]), std::hypot(test_case[0], test_case[1]));
    }
}

TEST(ErrorGammaFunctionsTest, ErrorFunctions)
{
    double values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Erf(val), std::erf(val));
        EXPECT_DOUBLE_EQ(Erfc(val), std::erfc(val));
    }
}

TEST(ErrorGammaFunctionsTest, GammaFunctions)
{
    double values[] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(tGmma(val), std::tgamma(val));
        EXPECT_DOUBLE_EQ(lGmma(val), std::lgamma(val));
    }
}

TEST(FloatingPointTest, RoundingFunctions)
{
    double values[] = {-2.7, -2.3, -1.5, -0.5, 0.0, 0.5, 1.5, 2.3, 2.7};

    for (double val : values)
    {
        EXPECT_DOUBLE_EQ(Ceil(val), std::ceil(val));
        EXPECT_DOUBLE_EQ(Floor(val), std::floor(val));
        EXPECT_DOUBLE_EQ(Trunc(val), std::trunc(val));
        EXPECT_DOUBLE_EQ(Round(val), std::round(val));
        EXPECT_DOUBLE_EQ(NearbyInt(val), std::nearbyint(val));
    }
}

TEST(FloatingPointTest, ManipulationFunctions)
{
    double val = 3.14159;
    int exp;

    double frac     = Frexp(val, &exp);
    double std_frac = std::frexp(val, &exp);
    EXPECT_DOUBLE_EQ(frac, std_frac);

    EXPECT_DOUBLE_EQ(Ldexp(0.5, 3), std::ldexp(0.5, 3));

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

        EXPECT_DOUBLE_EQ(frac_part, std_frac_part);
        EXPECT_DOUBLE_EQ(int_part, std_int_part);
    }
}

TEST(FloatingPointTest, NextAfterFunction)
{
    EXPECT_DOUBLE_EQ(NextAfter(1.0, 2.0), std::nextafter(1.0, 2.0));
    EXPECT_DOUBLE_EQ(NextAfter(1.0, 0.0), std::nextafter(1.0, 0.0));
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
        EXPECT_DOUBLE_EQ(CopySign(test_case[0], test_case[1]),
                         std::copysign(test_case[0], test_case[1]));
    }
}


TEST(FloatClassificationTest, IsFiniteTest)
{
    // 测试有限数值
    EXPECT_TRUE(IsFinite(0.0f));
    EXPECT_TRUE(IsFinite(-0.0f));
    EXPECT_TRUE(IsFinite(1.0f));
    EXPECT_TRUE(IsFinite(-1.0f));
    EXPECT_TRUE(IsFinite(3.14159f));
    EXPECT_TRUE(IsFinite(-2.71828f));
    EXPECT_TRUE(IsFinite(std::numeric_limits<f32>::min()));
    EXPECT_TRUE(IsFinite(std::numeric_limits<f32>::max()));
    EXPECT_TRUE(IsFinite(std::numeric_limits<f32>::denorm_min()));

    // 测试双精度
    EXPECT_TRUE(IsFinite(0.0));
    EXPECT_TRUE(IsFinite(1.0));
    EXPECT_TRUE(IsFinite(-1.0));
    EXPECT_TRUE(IsFinite(std::numeric_limits<f64>::min()));
    EXPECT_TRUE(IsFinite(std::numeric_limits<f64>::max()));

    // 测试无穷大
    EXPECT_FALSE(IsFinite(std::numeric_limits<f32>::infinity()));
    EXPECT_FALSE(IsFinite(-std::numeric_limits<f32>::infinity()));
    EXPECT_FALSE(IsFinite(std::numeric_limits<f64>::infinity()));
    EXPECT_FALSE(IsFinite(-std::numeric_limits<f64>::infinity()));

    // 测试 NaN
    EXPECT_FALSE(IsFinite(std::numeric_limits<f32>::quiet_NaN()));
    EXPECT_FALSE(IsFinite(std::numeric_limits<f32>::signaling_NaN()));
    EXPECT_FALSE(IsFinite(std::numeric_limits<f64>::quiet_NaN()));
    EXPECT_FALSE(IsFinite(std::numeric_limits<f64>::signaling_NaN()));
}

TEST(FloatClassificationTest, IsInfTest)
{
    // 测试无穷大
    EXPECT_TRUE(IsInf(std::numeric_limits<f32>::infinity()));
    EXPECT_TRUE(IsInf(-std::numeric_limits<f32>::infinity()));
    EXPECT_TRUE(IsInf(std::numeric_limits<f64>::infinity()));
    EXPECT_TRUE(IsInf(-std::numeric_limits<f64>::infinity()));

    // 测试有限数值
    EXPECT_FALSE(IsInf(0.0f));
    EXPECT_FALSE(IsInf(-0.0f));
    EXPECT_FALSE(IsInf(1.0f));
    EXPECT_FALSE(IsInf(-1.0f));
    EXPECT_FALSE(IsInf(std::numeric_limits<f32>::max()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f32>::min()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f64>::max()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f64>::min()));

    // 测试 NaN
    EXPECT_FALSE(IsInf(std::numeric_limits<f32>::quiet_NaN()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f32>::signaling_NaN()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f64>::quiet_NaN()));
    EXPECT_FALSE(IsInf(std::numeric_limits<f64>::signaling_NaN()));
}

TEST(FloatClassificationTest, IsNaNTest)
{
    // 测试 NaN
    EXPECT_TRUE(IsNaN(std::numeric_limits<f32>::quiet_NaN()));
    EXPECT_TRUE(IsNaN(std::numeric_limits<f32>::signaling_NaN()));
    EXPECT_TRUE(IsNaN(std::numeric_limits<f64>::quiet_NaN()));
    EXPECT_TRUE(IsNaN(std::numeric_limits<f64>::signaling_NaN()));

    // 测试有限数值
    EXPECT_FALSE(IsNaN(0.0f));
    EXPECT_FALSE(IsNaN(-0.0f));
    EXPECT_FALSE(IsNaN(1.0f));
    EXPECT_FALSE(IsNaN(-1.0f));
    EXPECT_FALSE(IsNaN(3.14159f));
    EXPECT_FALSE(IsNaN(-2.71828f));
    EXPECT_FALSE(IsNaN(std::numeric_limits<f32>::max()));
    EXPECT_FALSE(IsNaN(std::numeric_limits<f32>::min()));
    EXPECT_FALSE(IsNaN(std::numeric_limits<f64>::max()));
    EXPECT_FALSE(IsNaN(std::numeric_limits<f64>::min()));

    // 测试无穷大
    EXPECT_FALSE(IsNaN(std::numeric_limits<f32>::infinity()));
    EXPECT_FALSE(IsNaN(-std::numeric_limits<f32>::infinity()));
    EXPECT_FALSE(IsNaN(std::numeric_limits<f64>::infinity()));
    EXPECT_FALSE(IsNaN(-std::numeric_limits<f64>::infinity()));
}

TEST(FloatBitOperationsTest, FloatToBitsTest)
{
    // 测试单精度浮点数
    f32 f32_zero  = 0.0f;
    u32 bits_zero = FloatToBits(f32_zero);
    EXPECT_EQ(bits_zero, 0x00000000u);

    f32 f32_neg_zero  = -0.0f;
    u32 bits_neg_zero = FloatToBits(f32_neg_zero);
    EXPECT_EQ(bits_neg_zero, 0x80000000u);

    f32 f32_one  = 1.0f;
    u32 bits_one = FloatToBits(f32_one);
    EXPECT_EQ(bits_one, 0x3F800000u);

    f32 f32_neg_one  = -1.0f;
    u32 bits_neg_one = FloatToBits(f32_neg_one);
    EXPECT_EQ(bits_neg_one, 0xBF800000u);

    f32 f32_inf  = std::numeric_limits<f32>::infinity();
    u32 bits_inf = FloatToBits(f32_inf);
    EXPECT_EQ(bits_inf, 0x7F800000u);

    f32 f32_neg_inf  = -std::numeric_limits<f32>::infinity();
    u32 bits_neg_inf = FloatToBits(f32_neg_inf);
    EXPECT_EQ(bits_neg_inf, 0xFF800000u);

    // 测试双精度浮点数
    f64 f64_zero    = 0.0;
    u64 bits64_zero = FloatToBits(f64_zero);
    EXPECT_EQ(bits64_zero, 0x0000000000000000ull);

    f64 f64_one    = 1.0;
    u64 bits64_one = FloatToBits(f64_one);
    EXPECT_EQ(bits64_one, 0x3FF0000000000000ull);

    f64 f64_neg_one    = -1.0;
    u64 bits64_neg_one = FloatToBits(f64_neg_one);
    EXPECT_EQ(bits64_neg_one, 0xBFF0000000000000ull);
}

TEST(FloatBitOperationsTest, BitsToFloatTest)
{
    // 测试单精度浮点数
    u32 bits_zero = 0x00000000u;
    f32 f32_zero  = BitsToFloat(bits_zero);
    EXPECT_EQ(f32_zero, 0.0f);

    u32 bits_neg_zero = 0x80000000u;
    f32 f32_neg_zero  = BitsToFloat(bits_neg_zero);
    EXPECT_EQ(f32_neg_zero, -0.0f);

    u32 bits_one = 0x3F800000u;
    f32 f32_one  = BitsToFloat(bits_one);
    EXPECT_EQ(f32_one, 1.0f);

    u32 bits_neg_one = 0xBF800000u;
    f32 f32_neg_one  = BitsToFloat(bits_neg_one);
    EXPECT_EQ(f32_neg_one, -1.0f);

    u32 bits_inf = 0x7F800000u;
    f32 f32_inf  = BitsToFloat(bits_inf);
    EXPECT_TRUE(IsInf(f32_inf));
    EXPECT_GT(f32_inf, 0.0f);

    u32 bits_neg_inf = 0xFF800000u;
    f32 f32_neg_inf  = BitsToFloat(bits_neg_inf);
    EXPECT_TRUE(IsInf(f32_neg_inf));
    EXPECT_LT(f32_neg_inf, 0.0f);

    // 测试双精度浮点数
    u64 bits64_zero = 0x0000000000000000ull;
    f64 f64_zero    = BitsToFloat(bits64_zero);
    EXPECT_EQ(f64_zero, 0.0);

    u64 bits64_one = 0x3FF0000000000000ull;
    f64 f64_one    = BitsToFloat(bits64_one);
    EXPECT_EQ(f64_one, 1.0);

    u64 bits64_neg_one = 0xBFF0000000000000ull;
    f64 f64_neg_one    = BitsToFloat(bits64_neg_one);
    EXPECT_EQ(f64_neg_one, -1.0);
}

TEST(FloatBitOperationsTest, RoundTripTest)
{
    // 测试 FloatToBits 和 BitsToFloat 的往返转换
    std::vector<f32> test_values_f32 = {
            0.0f,
            -0.0f,
            1.0f,
            -1.0f,
            2.0f,
            -2.0f,
            3.14159f,
            -2.71828f,
            0.5f,
            -0.5f,
            std::numeric_limits<f32>::min(),
            std::numeric_limits<f32>::max(),
            std::numeric_limits<f32>::denorm_min(),
            std::numeric_limits<f32>::infinity(),
            -std::numeric_limits<f32>::infinity()
    };

    for (f32 value : test_values_f32)
    {
        u32 bits      = FloatToBits(value);
        f32 recovered = BitsToFloat(bits);
        if (IsNaN(value))
        {
            EXPECT_TRUE(IsNaN(recovered));
        }
        else
        {
            EXPECT_EQ(value, recovered);
        }
    }

    std::vector<f64> test_values_f64 = {
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.141592653589793,
            -2.718281828459045,
            std::numeric_limits<f64>::min(),
            std::numeric_limits<f64>::max(),
            std::numeric_limits<f64>::denorm_min(),
            std::numeric_limits<f64>::infinity(),
            -std::numeric_limits<f64>::infinity()
    };

    for (f64 value : test_values_f64)
    {
        u64 bits      = FloatToBits(value);
        f64 recovered = BitsToFloat(bits);
        if (IsNaN(value))
        {
            EXPECT_TRUE(IsNaN(recovered));
        }
        else
        {
            EXPECT_EQ(value, recovered);
        }
    }
}

TEST(FloatComponentsTest, ExponentTest)
{
    EXPECT_EQ(Exponent(1.0f), 0);    // 2^0
    EXPECT_EQ(Exponent(2.0f), 1);    // 2^1
    EXPECT_EQ(Exponent(4.0f), 2);    // 2^2
    EXPECT_EQ(Exponent(8.0f), 3);    // 2^3
    EXPECT_EQ(Exponent(0.5f), -1);   // 2^-1
    EXPECT_EQ(Exponent(0.25f), -2);  // 2^-2
    EXPECT_EQ(Exponent(0.125f), -3); // 2^-3

    EXPECT_EQ(Exponent(1.0), 0);   // 2^0
    EXPECT_EQ(Exponent(2.0), 1);   // 2^1
    EXPECT_EQ(Exponent(4.0), 2);   // 2^2
    EXPECT_EQ(Exponent(0.5), -1);  // 2^-1
    EXPECT_EQ(Exponent(0.25), -2); // 2^-2

    EXPECT_EQ(Exponent(std::numeric_limits<f32>::infinity()), 128);
    EXPECT_EQ(Exponent(-std::numeric_limits<f32>::infinity()), 128);
    EXPECT_EQ(Exponent(std::numeric_limits<f64>::infinity()), 1024);
    EXPECT_EQ(Exponent(-std::numeric_limits<f64>::infinity()), 1024);
}

TEST(FloatComponentsTest, SignificandTest)
{
    // 测试单精度浮点数尾数
    EXPECT_EQ(Significand(1.0f), 0u);          // 1.0 的尾数为 0
    EXPECT_EQ(Significand(1.5f), (1u << 22));  // 1.5 的尾数
    EXPECT_EQ(Significand(1.25f), (1u << 21)); // 1.25 的尾数

    // 测试双精度浮点数尾数
    EXPECT_EQ(Significand(1.0), 0ull);          // 1.0 的尾数为 0
    EXPECT_EQ(Significand(1.5), (1ull << 51));  // 1.5 的尾数
    EXPECT_EQ(Significand(1.25), (1ull << 50)); // 1.25 的尾数

    // 测试零值
    EXPECT_EQ(Significand(0.0f), 0u);
    EXPECT_EQ(Significand(-0.0f), 0u);
    EXPECT_EQ(Significand(0.0), 0ull);
    EXPECT_EQ(Significand(-0.0), 0ull);
}

TEST(FloatComponentsTest, SignBitTest)
{
    // 测试单精度浮点数符号位
    EXPECT_EQ(SignBit(1.0f), 0u);
    EXPECT_EQ(SignBit(-1.0f), 0x80000000u);
    EXPECT_EQ(SignBit(0.0f), 0u);
    EXPECT_EQ(SignBit(-0.0f), 0x80000000u);
    EXPECT_EQ(SignBit(std::numeric_limits<f32>::infinity()), 0u);
    EXPECT_EQ(SignBit(-std::numeric_limits<f32>::infinity()), 0x80000000u);

    // 测试双精度浮点数符号位
    EXPECT_EQ(SignBit(1.0), 0ull);
    EXPECT_EQ(SignBit(-1.0), 0x8000000000000000ull);
    EXPECT_EQ(SignBit(0.0), 0ull);
    EXPECT_EQ(SignBit(-0.0), 0x8000000000000000ull);
    EXPECT_EQ(SignBit(std::numeric_limits<f64>::infinity()), 0ull);
    EXPECT_EQ(SignBit(-std::numeric_limits<f64>::infinity()), 0x8000000000000000ull);
}

TEST(FloatPrecisionTest, NextFloatUpTest)
{
    // 测试正数
    f32 x    = 1.0f;
    f32 next = NextFloatUp(x);
    EXPECT_GT(next, x);
    EXPECT_LT(next - x, 1e-6f);

    // 测试负数
    f32 neg_x    = -1.0f;
    f32 next_neg = NextFloatUp(neg_x);
    EXPECT_GT(next_neg, neg_x);

    // 测试零值
    f32 zero      = 0.0f;
    f32 next_zero = NextFloatUp(zero);
    EXPECT_GT(next_zero, zero);
    EXPECT_EQ(next_zero, std::numeric_limits<f32>::denorm_min());

    f32 neg_zero      = -0.0f;
    f32 next_neg_zero = NextFloatUp(neg_zero);
    EXPECT_GT(next_neg_zero, neg_zero);
    EXPECT_EQ(next_neg_zero, std::numeric_limits<f32>::denorm_min());

    // 测试无穷大
    f32 inf      = std::numeric_limits<f32>::infinity();
    f32 next_inf = NextFloatUp(inf);
    EXPECT_EQ(next_inf, inf);

    f32 neg_inf      = -std::numeric_limits<f32>::infinity();
    f32 next_neg_inf = NextFloatUp(neg_inf);
    EXPECT_GT(next_neg_inf, neg_inf);

    // 测试双精度
    f64 x64    = 1.0;
    f64 next64 = NextFloatUp(x64);
    EXPECT_GT(next64, x64);
    EXPECT_LT(next64 - x64, 1e-15);
}

TEST(FloatPrecisionTest, NextFloatDownTest)
{
    // 测试正数
    f32 x    = 1.0f;
    f32 prev = NextFloatDown(x);
    EXPECT_LT(prev, x);
    EXPECT_LT(x - prev, 1e-6f);

    // 测试负数
    f32 neg_x    = -1.0f;
    f32 prev_neg = NextFloatDown(neg_x);
    EXPECT_LT(prev_neg, neg_x);

    // 测试零值
    f32 zero      = 0.0f;
    f32 prev_zero = NextFloatDown(zero);
    EXPECT_LT(prev_zero, zero);
    EXPECT_EQ(prev_zero, -std::numeric_limits<f32>::denorm_min());

    // 测试无穷大
    f32 inf      = std::numeric_limits<f32>::infinity();
    f32 prev_inf = NextFloatDown(inf);
    EXPECT_LT(prev_inf, inf);

    f32 neg_inf      = -std::numeric_limits<f32>::infinity();
    f32 prev_neg_inf = NextFloatDown(neg_inf);
    EXPECT_EQ(prev_neg_inf, neg_inf);

    // 测试双精度
    f64 x64    = 1.0;
    f64 prev64 = NextFloatDown(x64);
    EXPECT_LT(prev64, x64);
    EXPECT_LT(x64 - prev64, 1e-15);
}

TEST(FloatPrecisionTest, NextFloatConsistencyTest)
{
    // 测试 NextFloatUp 和 NextFloatDown 的一致性
    std::vector<f32> test_values = {
            1.0f,
            -1.0f,
            2.0f,
            -2.0f,
            0.5f,
            -0.5f,
            100.0f,
            -100.0f,
            0.001f,
            -0.001f
    };

    for (f32 value : test_values)
    {
        f32 up   = NextFloatUp(value);
        f32 down = NextFloatDown(up);
        EXPECT_EQ(down, value) << "Value: " << value;

        f32 down_first = NextFloatDown(value);
        f32 up_again   = NextFloatUp(down_first);
        EXPECT_EQ(up_again, value) << "Value: " << value;
    }
}

TEST(RoundingOperationsTest, AddRoundingTest)
{
    f32 a = 1.0f;
    f32 b = std::numeric_limits<f32>::epsilon();

    f32 sum_up     = AddRoundUp(a, b);
    f32 sum_down   = AddRoundDown(a, b);
    f32 sum_normal = a + b;

    EXPECT_GE(sum_up, sum_normal);
    EXPECT_LE(sum_down, sum_normal);
    EXPECT_GE(sum_up, sum_down);

    // 测试双精度
    f64 a64 = 1.0;
    f64 b64 = std::numeric_limits<f64>::epsilon();

    f64 sum_up64     = AddRoundUp(a64, b64);
    f64 sum_down64   = AddRoundDown(a64, b64);
    f64 sum_normal64 = a64 + b64;

    EXPECT_GE(sum_up64, sum_normal64);
    EXPECT_LE(sum_down64, sum_normal64);
}

TEST(RoundingOperationsTest, SubRoundingTest)
{
    f32 a = 1.0f;
    f32 b = std::numeric_limits<f32>::epsilon();

    f32 diff_up     = SubRoundUp(a, b);
    f32 diff_down   = SubRoundDown(a, b);
    f32 diff_normal = a - b;

    EXPECT_GE(diff_up, diff_normal);
    EXPECT_LE(diff_down, diff_normal);
    EXPECT_GE(diff_up, diff_down);
}

TEST(RoundingOperationsTest, MulRoundingTest)
{
    f32 a = 1.0f + std::numeric_limits<f32>::epsilon();
    f32 b = 1.0f + std::numeric_limits<f32>::epsilon();

    f32 prod_up     = MulRoundUp(a, b);
    f32 prod_down   = MulRoundDown(a, b);
    f32 prod_normal = a * b;

    EXPECT_GE(prod_up, prod_normal);
    EXPECT_LE(prod_down, prod_normal);
    EXPECT_GE(prod_up, prod_down);
}

TEST(RoundingOperationsTest, DivRoundingTest)
{
    f32 a = 1.0f;
    f32 b = 3.0f;

    f32 quot_up     = DivRoundUp(a, b);
    f32 quot_down   = DivRoundDown(a, b);
    f32 quot_normal = a / b;

    EXPECT_GE(quot_up, quot_normal);
    EXPECT_LE(quot_down, quot_normal);
    EXPECT_GE(quot_up, quot_down);
}

TEST(RoundingOperationsTest, SqrtRoundingTest)
{
    f32 x = 2.0f;

    f32 sqrt_up     = SqrtRoundUp(x);
    f32 sqrt_down   = SqrtRoundDown(x);
    f32 sqrt_normal = std::sqrt(x);

    EXPECT_GE(sqrt_up, sqrt_normal);
    EXPECT_LE(sqrt_down, sqrt_normal);
    EXPECT_GE(sqrt_up, sqrt_down);
}

TEST(RoundingOperationsTest, FMARoundingTest)
{
    f32 a = 1.0f + std::numeric_limits<f32>::epsilon();
    f32 b = 1.0f + std::numeric_limits<f32>::epsilon();
    f32 c = 1.0f + std::numeric_limits<f32>::epsilon();

    f32 fma_up     = FMARoundUp(a, b, c);
    f32 fma_down   = FMARoundDown(a, b, c);
    f32 fma_normal = std::fma(a, b, c);

    EXPECT_GE(fma_up, fma_normal);
    EXPECT_LE(fma_down, fma_normal);
    EXPECT_GE(fma_up, fma_down);
}

TEST(GammaFunctionTest, GammaTest)
{
    // 测试一些基本值
    EXPECT_GT(Gamma(1), 0.0);
    EXPECT_GT(Gamma(2), Gamma(1));
    EXPECT_GT(Gamma(3), Gamma(2));

    // 测试负值
    EXPECT_LT(Gamma(-1), 0.0);
    EXPECT_LT(Gamma(-2), Gamma(-1));

    // 测试零值
    EXPECT_EQ(Gamma(0), 0.0);
}

TEST(EdgeCasesTest, InfinityHandlingTest)
{
    f32 inf     = std::numeric_limits<f32>::infinity();
    f32 neg_inf = -std::numeric_limits<f32>::infinity();

    // 测试无穷大的分类
    EXPECT_TRUE(IsInf(inf));
    EXPECT_TRUE(IsInf(neg_inf));
    EXPECT_FALSE(IsFinite(inf));
    EXPECT_FALSE(IsFinite(neg_inf));
    EXPECT_FALSE(IsNaN(inf));
    EXPECT_FALSE(IsNaN(neg_inf));

    // 测试无穷大的位操作
    u32 inf_bits     = FloatToBits(inf);
    u32 neg_inf_bits = FloatToBits(neg_inf);
    EXPECT_EQ(inf_bits, 0x7F800000u);
    EXPECT_EQ(neg_inf_bits, 0xFF800000u);

    f32 recovered_inf     = BitsToFloat(inf_bits);
    f32 recovered_neg_inf = BitsToFloat(neg_inf_bits);
    EXPECT_TRUE(IsInf(recovered_inf));
    EXPECT_TRUE(IsInf(recovered_neg_inf));
    EXPECT_GT(recovered_inf, 0.0f);
    EXPECT_LT(recovered_neg_inf, 0.0f);
}

TEST(EdgeCasesTest, NaNHandlingTest)
{
    f32 nan  = std::numeric_limits<f32>::quiet_NaN();
    f32 snan = std::numeric_limits<f32>::signaling_NaN();

    // 测试 NaN 的分类
    EXPECT_TRUE(IsNaN(nan));
    EXPECT_TRUE(IsNaN(snan));
    EXPECT_FALSE(IsFinite(nan));
    EXPECT_FALSE(IsFinite(snan));
    EXPECT_FALSE(IsInf(nan));
    EXPECT_FALSE(IsInf(snan));

    // 测试 NaN 的位操作
    u32 nan_bits      = FloatToBits(nan);
    f32 recovered_nan = BitsToFloat(nan_bits);
    EXPECT_TRUE(IsNaN(recovered_nan));
}

TEST(EdgeCasesTest, ZeroHandlingTest)
{
    f32 pos_zero = 0.0f;
    f32 neg_zero = -0.0f;

    // 测试零值的分类
    EXPECT_TRUE(IsFinite(pos_zero));
    EXPECT_TRUE(IsFinite(neg_zero));
    EXPECT_FALSE(IsInf(pos_zero));
    EXPECT_FALSE(IsInf(neg_zero));
    EXPECT_FALSE(IsNaN(pos_zero));
    EXPECT_FALSE(IsNaN(neg_zero));

    // 测试零值的位操作
    u32 pos_zero_bits = FloatToBits(pos_zero);
    u32 neg_zero_bits = FloatToBits(neg_zero);
    EXPECT_EQ(pos_zero_bits, 0x00000000u);
    EXPECT_EQ(neg_zero_bits, 0x80000000u);

    // 测试零值的符号位
    EXPECT_EQ(SignBit(pos_zero), 0u);
    EXPECT_EQ(SignBit(neg_zero), 0x80000000u);

    // 测试零值的精度控制
    f32 next_up_pos   = NextFloatUp(pos_zero);
    f32 next_up_neg   = NextFloatUp(neg_zero);
    f32 next_down_pos = NextFloatDown(pos_zero);

    EXPECT_GT(next_up_pos, pos_zero);
    EXPECT_GT(next_up_neg, neg_zero);
    EXPECT_LT(next_down_pos, pos_zero);
    EXPECT_EQ(next_up_pos, std::numeric_limits<f32>::denorm_min());
    EXPECT_EQ(next_up_neg, std::numeric_limits<f32>::denorm_min());
    EXPECT_EQ(next_down_pos, -std::numeric_limits<f32>::denorm_min());
}

TEST(EdgeCasesTest, DenormalHandlingTest)
{
    f32 denorm_min     = std::numeric_limits<f32>::denorm_min();
    f32 neg_denorm_min = -std::numeric_limits<f32>::denorm_min();

    // 测试非规格化数的分类
    EXPECT_TRUE(IsFinite(denorm_min));
    EXPECT_TRUE(IsFinite(neg_denorm_min));
    EXPECT_FALSE(IsInf(denorm_min));
    EXPECT_FALSE(IsInf(neg_denorm_min));
    EXPECT_FALSE(IsNaN(denorm_min));
    EXPECT_FALSE(IsNaN(neg_denorm_min));

    // 测试非规格化数的位操作
    u32 denorm_bits     = FloatToBits(denorm_min);
    u32 neg_denorm_bits = FloatToBits(neg_denorm_min);
    EXPECT_EQ(denorm_bits, 0x00000001u);
    EXPECT_EQ(neg_denorm_bits, 0x80000001u);

    f32 recovered_denorm     = BitsToFloat(denorm_bits);
    f32 recovered_neg_denorm = BitsToFloat(neg_denorm_bits);
    EXPECT_EQ(recovered_denorm, denorm_min);
    EXPECT_EQ(recovered_neg_denorm, neg_denorm_min);
}

TEST(EdgeCasesTest, ExtremeValuesTest)
{
    f32 max_val = std::numeric_limits<f32>::max();
    f32 min_val = std::numeric_limits<f32>::min();

    // 测试极值的分类
    EXPECT_TRUE(IsFinite(max_val));
    EXPECT_TRUE(IsFinite(min_val));
    EXPECT_FALSE(IsInf(max_val));
    EXPECT_FALSE(IsInf(min_val));
    EXPECT_FALSE(IsNaN(max_val));
    EXPECT_FALSE(IsNaN(min_val));

    // 测试极值的位操作往返转换
    u32 max_bits      = FloatToBits(max_val);
    u32 min_bits      = FloatToBits(min_val);
    f32 recovered_max = BitsToFloat(max_bits);
    f32 recovered_min = BitsToFloat(min_bits);
    EXPECT_EQ(recovered_max, max_val);
    EXPECT_EQ(recovered_min, min_val);

    // 测试极值的精度控制
    f32 next_up_max = NextFloatUp(max_val);
    EXPECT_TRUE(IsInf(next_up_max));

    f32 next_down_min = NextFloatDown(min_val);
    EXPECT_LT(next_down_min, min_val);
}


TEST(BasicFunctionsTest, ModFunction)
{
    EXPECT_EQ(Mod(7, 3), 1);
    EXPECT_EQ(Mod(8, 3), 2);
    EXPECT_EQ(Mod(9, 3), 0);
    EXPECT_EQ(Mod(10, 3), 1);

    EXPECT_EQ(Mod(-7, 3), 2); // 负数取模应该返回正值
    EXPECT_EQ(Mod(-8, 3), 1);
    EXPECT_EQ(Mod(-9, 3), 0);

    EXPECT_EQ(Mod(0, 5), 0);
    EXPECT_EQ(Mod(1, 5), 1);
    EXPECT_EQ(Mod(4, 5), 4);
    EXPECT_EQ(Mod(5, 5), 0);
}

TEST(PowerFunctionsTest, InvHypotFunction)
{
    double test_cases[][2] = {
            {3.0, 4.0},
            {5.0, 12.0},
            {8.0, 15.0},
            {1.0, 1.0},
            {0.0, 1.0},
            {1.0, 0.0}
    };

    for (auto& test_case : test_cases)
    {
        double expected = 1.0 / std::hypot(test_case[0], test_case[1]);
        EXPECT_DOUBLE_EQ(InvHypot(test_case[0], test_case[1]), expected);
    }
}

TEST(UtilityFunctionsTest, AlmostZeroFunction)
{
    EXPECT_TRUE(AlmostZero(0.0f));
    EXPECT_TRUE(AlmostZero(0.0));

    // 非常小的值应该被认为是"几乎为零"
    EXPECT_TRUE(AlmostZero(1e-8f));
    EXPECT_TRUE(AlmostZero(1e-15));

    // 较大的值不应该被认为是"几乎为零"
    EXPECT_FALSE(AlmostZero(0.1f));
    EXPECT_FALSE(AlmostZero(0.01f));
    EXPECT_FALSE(AlmostZero(1.0f));
    EXPECT_FALSE(AlmostZero(-0.1f));
}

TEST(UtilityFunctionsTest, SincFunction)
{
    // sinc(0) = 1
    EXPECT_DOUBLE_EQ(Sinc(0.0), 1.0);
    EXPECT_FLOAT_EQ(Sinc(0.0f), 1.0f);

    // sinc(π) = 0
    EXPECT_NEAR(Sinc(kPi), 0, kEpsilonD);
    EXPECT_NEAR(Sinc(static_cast<float>(kPi)), 0, kEpsilonF);

    // sinc(π/2) = 2/π
    double expected_half_pi = 2.0 / kPi;
    EXPECT_DOUBLE_EQ(Sinc(kPi / 2), expected_half_pi);

    // 测试一些其他值
    EXPECT_DOUBLE_EQ(Sinc(1.0), std::sin(1.0) / 1.0);
    EXPECT_DOUBLE_EQ(Sinc(2.0), std::sin(2.0) / 2.0);
    EXPECT_DOUBLE_EQ(Sinc(-1.0), std::sin(-1.0) / (-1.0));
}

TEST(UtilityFunctionsTest, SqrFunction)
{
    EXPECT_EQ(Sqr(0), 0);
    EXPECT_EQ(Sqr(1), 1);
    EXPECT_EQ(Sqr(2), 4);
    EXPECT_EQ(Sqr(3), 9);
    EXPECT_EQ(Sqr(-2), 4);
    EXPECT_EQ(Sqr(-3), 9);

    EXPECT_DOUBLE_EQ(Sqr(3.14), 3.14 * 3.14);
    EXPECT_DOUBLE_EQ(Sqr(-2.5), 6.25);
    EXPECT_FLOAT_EQ(Sqr(1.5f), 2.25f);
}

TEST(ConversionFunctionsTest, AngleConversions)
{
    // 度数转弧度
    EXPECT_DOUBLE_EQ(Degree2Radian(0.0), 0.0);
    EXPECT_DOUBLE_EQ(Degree2Radian(90.0), kPi / 2);
    EXPECT_DOUBLE_EQ(Degree2Radian(180.0), kPi);
    EXPECT_DOUBLE_EQ(Degree2Radian(270.0), 3 * kPi / 2);
    EXPECT_DOUBLE_EQ(Degree2Radian(360.0), 2 * kPi);
    EXPECT_DOUBLE_EQ(Degree2Radian(-90.0), -kPi / 2);

    // 弧度转度数
    EXPECT_DOUBLE_EQ(Radian2Degree(0.0), 0.0);
    EXPECT_DOUBLE_EQ(Radian2Degree(kPi / 2), 90.0);
    EXPECT_DOUBLE_EQ(Radian2Degree(kPi), 180.0);
    EXPECT_DOUBLE_EQ(Radian2Degree(3 * kPi / 2), 270.0);
    EXPECT_DOUBLE_EQ(Radian2Degree(2 * kPi), 360.0);
    EXPECT_DOUBLE_EQ(Radian2Degree(-kPi / 2), -90.0);

    // 往返转换测试
    double degrees[] = {0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 270.0, 360.0};
    for (double deg : degrees)
    {
        double rad         = Degree2Radian(deg);
        double back_to_deg = Radian2Degree(rad);
        EXPECT_DOUBLE_EQ(back_to_deg, deg);
    }
}

TEST(UtilityFunctionsTest, LerpFunction)
{
    // 基本线性插值测试
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 1.0), 10.0);
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 0.5), 5.0);
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 0.25), 2.5);
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 0.75), 7.5);

    // 负值测试
    EXPECT_DOUBLE_EQ(Lerp(-5.0, 5.0, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(Lerp(-10.0, -5.0, 0.5), -7.5);

    // 超出范围测试（外推）
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, -0.5), -5.0);
    EXPECT_DOUBLE_EQ(Lerp(0.0, 10.0, 1.5), 15.0);

    // 混合类型测试
    EXPECT_NEAR(Lerp(0, 10.0, 0.3f), 3.0, kEpsilonF);
    EXPECT_FLOAT_EQ(Lerp(0.0f, 5, 0.4), 2.0f);
}

TEST(UtilityFunctionsTest, ClampFunction)
{
    // 基本限制测试
    EXPECT_EQ(Clamp(5, 0, 10), 5);
    EXPECT_EQ(Clamp(-5, 0, 10), 0);
    EXPECT_EQ(Clamp(15, 0, 10), 10);
    EXPECT_EQ(Clamp(0, 0, 10), 0);
    EXPECT_EQ(Clamp(10, 0, 10), 10);

    // 浮点数测试
    EXPECT_DOUBLE_EQ(Clamp(3.14, 0.0, 5.0), 3.14);
    EXPECT_DOUBLE_EQ(Clamp(-1.5, 0.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(Clamp(7.8, 0.0, 5.0), 5.0);

    // 负范围测试
    EXPECT_EQ(Clamp(-3, -10, -1), -3);
    EXPECT_EQ(Clamp(-15, -10, -1), -10);
    EXPECT_EQ(Clamp(5, -10, -1), -1);
}

TEST(UtilityFunctionsTest, Clamp01Function)
{
    // 基本 0-1 限制测试
    EXPECT_DOUBLE_EQ(Clamp01(0.5), 0.5);
    EXPECT_DOUBLE_EQ(Clamp01(0.0), 0.0);
    EXPECT_DOUBLE_EQ(Clamp01(1.0), 1.0);
    EXPECT_DOUBLE_EQ(Clamp01(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(Clamp01(1.5), 1.0);
    EXPECT_DOUBLE_EQ(Clamp01(2.0), 1.0);
    EXPECT_DOUBLE_EQ(Clamp01(-1.0), 0.0);

    // 浮点数精度测试
    EXPECT_FLOAT_EQ(Clamp01(0.3f), 0.3f);
    EXPECT_FLOAT_EQ(Clamp01(-0.1f), 0.0f);
    EXPECT_FLOAT_EQ(Clamp01(1.1f), 1.0f);
}


TEST(EvaluatePolynomialTest, ConstantPolynomial)
{
    // P(x) = 5
    EXPECT_FLOAT_EQ(EvaluatePolynomial(2.0f, 5.0f), 5.0f);
    EXPECT_DOUBLE_EQ(EvaluatePolynomial(3.0, 7.0), 7.0);

    // Test with different types
    EXPECT_FLOAT_EQ(EvaluatePolynomial(1.5f, 42), 42.0f);
    EXPECT_DOUBLE_EQ(EvaluatePolynomial(2.5, 3.14), 3.14);
}

TEST(EvaluatePolynomialTest, LinearPolynomial)
{
    // P(x) = 3x + 2
    float x1      = 2.0f;
    float result1 = EvaluatePolynomial(x1, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(result1, 3.0f * x1 + 2.0f);

    // P(x) = -x + 5
    double x2      = 4.0;
    double result2 = EvaluatePolynomial(x2, 5.0, -1.0);
    EXPECT_DOUBLE_EQ(result2, -x2 + 5.0);

    // Test zero coefficient
    EXPECT_FLOAT_EQ(EvaluatePolynomial(3.0f, 0.0f, 2.0f), 6.0f);
}

TEST(EvaluatePolynomialTest, QuadraticPolynomial)
{
    // P(x) = x^2 + 2x + 1 = (x + 1)^2
    float x        = 3.0f;
    float result   = EvaluatePolynomial(x, 1.0f, 2.0f, 1.0f);
    float expected = x * x + 2.0f * x + 1.0f;
    EXPECT_FLOAT_EQ(result, expected);

    // P(x) = 2x^2 - 3x + 1
    double x2        = 2.5;
    double result2   = EvaluatePolynomial(x2, 1.0, -3.0, 2.0);
    double expected2 = 2.0 * x2 * x2 - 3.0 * x2 + 1.0;
    EXPECT_DOUBLE_EQ(result2, expected2);
}

TEST(EvaluatePolynomialTest, HighOrderPolynomial)
{
    // P(x) = x^4 + 2x^3 - x^2 + 3x - 5
    float x      = 1.5f;
    float result = EvaluatePolynomial(x, -5.0f, 3.0f, -1.0f, 2.0f, 1.0f);

    float expected = 1.0f * x * x * x * x +
            2.0f * x * x * x -
            1.0f * x * x +
            3.0f * x - 5.0f;
    EXPECT_FLOAT_EQ(result, expected);
}

TEST(EvaluatePolynomialTest, ZeroInput)
{
    // Test with x = 0
    EXPECT_FLOAT_EQ(EvaluatePolynomial(0.0f, 5.0f), 5.0f);
    EXPECT_FLOAT_EQ(EvaluatePolynomial(0.0f, 3.0f, 2.0f, 1.0f), 3.0f);
}

TEST(EvaluatePolynomialTest, NegativeInput)
{
    // Test with negative x
    float x        = -2.0f;
    float result   = EvaluatePolynomial(x, 1.0f, 2.0f, 3.0f);
    float expected = 3.0f * x * x + 2.0f * x + 1.0f;
    EXPECT_FLOAT_EQ(result, expected);
}

TEST(PolynomialEdgeCasesTest, QuadraticNumericalStability)
{
    // Test quadratic solver with coefficients that might cause numerical issues
    // Use a more reasonable test case: x^2 - 1000*x + 1 = 0
    // Roots are approximately 999.999 and 0.001
    double t0, t1;
    bool hasRoots = Quadratic(1.0, -1000.0, 1.0, &t0, &t1);

    EXPECT_TRUE(hasRoots);
    EXPECT_GT(t1, t0); // t1 should be the larger root

    // Verify the roots satisfy the equation with appropriate tolerance
    double residual0 = 1.0 * t0 * t0 - 1000.0 * t0 + 1.0;
    double residual1 = 1.0 * t1 * t1 - 1000.0 * t1 + 1.0;

    EXPECT_NEAR(residual0, 0.0, 1e-9);
    EXPECT_NEAR(residual1, 0.0, 1e-9);

    // Additional verification: product and sum of roots
    EXPECT_NEAR(t0 * t1, 1.0, 1e-12);    // Product of roots = c/a = 1
    EXPECT_NEAR(t0 + t1, 1000.0, 1e-10); // Sum of roots = -b/a = 1000
}

namespace
{
    bool IsApproximatelyEqual(f32 a, f32 b, f32 tolerance = 1e-3f)
    {
        if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f)
        {
            return std::abs(a - b) < tolerance;
        }
        return std::abs((a - b) / std::max(std::abs(a), std::abs(b))) < tolerance;
    }
} // namespace 

TEST(FastMathTest, FastExpTest)
{
    std::vector<f32> test_values = {
            0.0f,
            1.0f,
            -1.0f,
            2.0f,
            -2.0f,
            0.5f,
            -0.5f,
            3.0f,
            -3.0f,
            10.0f,
            -10.0f
    };

    for (f32 x : test_values)
    {
        f32 fast_result = FastExp(x);
        f32 std_result  = std::exp(x);

        // 快速指数函数应该在合理的误差范围内
        if (std::isfinite(std_result) && std_result > 0)
        {
            EXPECT_TRUE(IsApproximatelyEqual(fast_result, std_result, 0.1f))
                << "FastExp(" << x << ") = " << fast_result
                << ", std::exp(" << x << ") = " << std_result;
        }
    }
}

TEST(FastMathTest, FastSqrtTest)
{
    std::vector<f32> test_values = {
            1.0f,
            4.0f,
            9.0f,
            16.0f,
            25.0f,
            0.25f,
            0.5f,
            2.0f,
            10.0f,
            100.0f
    };

    for (f32 x : test_values)
    {
        f32 fast_result = FastSqrt(x);
        f32 std_result  = std::sqrt(x);

        // 快速平方根应该在合理的误差范围内
        EXPECT_TRUE(IsApproximatelyEqual(fast_result, std_result, 0.01f))
            << "FastSqrt(" << x << ") = " << fast_result
            << ", std::sqrt(" << x << ") = " << std_result;
    }
}

TEST(FastMathTest, FastCbrtTest)
{
    std::vector<f32> test_values = {
            1.0f,
            8.0f,
            27.0f,
            64.0f,
            125.0f,
            0.125f,
            0.5f,
            2.0f,
            10.0f,
            100.0f
    };

    for (f32 x : test_values)
    {
        f32 fast_result = FastCbrt(x);
        f32 std_result  = std::cbrt(x);

        // 快速立方根应该在合理的误差范围内
        EXPECT_TRUE(IsApproximatelyEqual(fast_result, std_result, 0.01f))
            << "FastCbrt(" << x << ") = " << fast_result
            << ", std::cbrt(" << x << ") = " << std_result;
    }
}

TEST(FastMathTest, FastInvSqrtTest)
{
    std::vector<f32> test_values_f32 = {
            1.0f,
            4.0f,
            9.0f,
            16.0f,
            25.0f,
            0.25f,
            0.5f,
            2.0f,
            10.0f,
            100.0f
    };

    for (f32 x : test_values_f32)
    {
        f32 fast_result = FastInvSqrt(x);
        f32 std_result  = 1.0f / std::sqrt(x);

        // 快速逆平方根应该在合理的误差范围内
        EXPECT_TRUE(IsApproximatelyEqual(fast_result, std_result, 0.01f))
            << "FastInvSqrt(" << x << ") = " << fast_result
            << ", 1/sqrt(" << x << ") = " << std_result;
    }

    // 测试双精度
    std::vector<f64> test_values_f64 = {
            1.0,
            4.0,
            9.0,
            16.0,
            25.0,
            0.25,
            0.5,
            2.0,
            10.0,
            100.0
    };

    for (f64 x : test_values_f64)
    {
        f64 fast_result = FastInvSqrt(x);
        f64 std_result  = 1.0 / std::sqrt(x);

        // 双精度快速逆平方根应该在合理的误差范围内
        EXPECT_TRUE(std::abs((fast_result - std_result) / std_result) < 0.01)
            << "FastInvSqrt(" << x << ") = " << fast_result
            << ", 1/sqrt(" << x << ") = " << std_result;
    }
}

TEST(AdvancedMathTest, GaussianFunction)
{
    // 标准正态分布 (mu=0, sigma=1)
    double std_normal_peak = Gaussian(0.0, 0.0, 1.0);
    double expected_peak   = 1.0 / std::sqrt(2 * kPi);
    EXPECT_DOUBLE_EQ(std_normal_peak, expected_peak);

    // 对称性测试
    EXPECT_DOUBLE_EQ(Gaussian(1.0, 0.0, 1.0), Gaussian(-1.0, 0.0, 1.0));
    EXPECT_DOUBLE_EQ(Gaussian(2.0, 0.0, 1.0), Gaussian(-2.0, 0.0, 1.0));

    // 不同参数测试
    double val1 = Gaussian(1.0, 1.0, 1.0); // 在均值处
    double val2 = Gaussian(0.0, 1.0, 1.0); // 偏离均值
    EXPECT_GT(val1, val2);                 // 在均值处应该更大

    // sigma 影响测试
    double narrow = Gaussian(0.0, 0.0, 0.5); // 较小的 sigma
    double wide   = Gaussian(0.0, 0.0, 2.0); // 较大的 sigma
    EXPECT_GT(narrow, wide);                 // 较小的 sigma 在峰值处应该更高
}

TEST(AdvancedMathTest, GaussianIntegralFunction)
{
    // 整个分布的积分应该接近 1（对于足够大的范围）
    double full_integral = GaussianIntegral(-5.0, 5.0, 0.0, 1.0);
    EXPECT_NEAR(full_integral, 1.0, kEpsilonF);

    // 对称性测试
    double left_half  = GaussianIntegral(-5.0, 0.0, 0.0, 1.0);
    double right_half = GaussianIntegral(0.0, 5.0, 0.0, 1.0);
    EXPECT_DOUBLE_EQ(left_half, right_half);
    EXPECT_NEAR(left_half + right_half, 1.0, kEpsilonF);

    // 单点积分应该为 0
    EXPECT_DOUBLE_EQ(GaussianIntegral(1.0, 1.0, 0.0, 1.0), 0.0);

    // 反向积分应该为负值
    double forward  = GaussianIntegral(0.0, 1.0, 0.0, 1.0);
    double backward = GaussianIntegral(1.0, 0.0, 0.0, 1.0);
    EXPECT_DOUBLE_EQ(forward, -backward);
}

TEST(NumberTheoryTest, GCDFunction)
{
    // 基本 GCD 测试
    EXPECT_EQ(GCD(12, 8), 4);
    EXPECT_EQ(GCD(8, 12), 4);
    EXPECT_EQ(GCD(17, 13), 1); // 互质数
    EXPECT_EQ(GCD(100, 25), 25);
    EXPECT_EQ(GCD(0, 5), 5);
    EXPECT_EQ(GCD(5, 0), 5);

    // 负数测试
    EXPECT_EQ(GCD(-12, 8), 4);
    EXPECT_EQ(GCD(12, -8), 4);
    EXPECT_EQ(GCD(-12, -8), 4);

    // 相同数字
    EXPECT_EQ(GCD(7, 7), 7);

    // 可变参数版本测试
    EXPECT_EQ(GCD(12, 8, 4), 4);
    EXPECT_EQ(GCD(60, 48, 36), 12);
    EXPECT_EQ(GCD(14, 21, 35), 7);
    EXPECT_EQ(GCD(10, 15, 20, 25), 5);
}

TEST(NumberTheoryTest, LCMFunction)
{
    // 基本 LCM 测试
    EXPECT_EQ(LCM(4, 6), 12);
    EXPECT_EQ(LCM(6, 4), 12);
    EXPECT_EQ(LCM(7, 5), 35); // 互质数
    EXPECT_EQ(LCM(12, 18), 36);
    EXPECT_EQ(LCM(1, 5), 5);

    // 相同数字
    EXPECT_EQ(LCM(8, 8), 8);

    // 可变参数版本测试
    EXPECT_EQ(LCM(4, 6, 8), 24);
    EXPECT_EQ(LCM(2, 3, 4), 12);
    EXPECT_EQ(LCM(6, 10, 15), 30);
    EXPECT_EQ(LCM(2, 4, 8, 16), 16);
}

TEST(NumberTheoryTest, GCDLCMRelationship)
{
    // 验证 GCD 和 LCM 的数学关系: GCD(a,b) * LCM(a,b) = a * b
    std::vector<std::pair<int, int>> test_pairs = {
            {12, 8},
            {15, 25},
            {7, 11},
            {24, 36},
            {13, 17}
    };

    for (const auto& [a, b] : test_pairs)
    {
        EXPECT_EQ(GCD(a, b) * LCM(a, b), a * b);
    }
}

TEST(UtilityFunctionsTest, MidpointFunction)
{
    // 整数中点测试
    EXPECT_DOUBLE_EQ(Midpoint(0, 10), 5.0);
    EXPECT_DOUBLE_EQ(Midpoint(10, 0), 5.0);
    EXPECT_DOUBLE_EQ(Midpoint(-5, 5), 0.0);
    EXPECT_DOUBLE_EQ(Midpoint(3, 7), 5.0);

    // 浮点数中点测试
    EXPECT_DOUBLE_EQ(Midpoint(1.0, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(Midpoint(-2.5, 2.5), 0.0);
    EXPECT_DOUBLE_EQ(Midpoint(0.0, 1.0), 0.5);

    // 混合类型测试
    EXPECT_DOUBLE_EQ(Midpoint(1, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(Midpoint(2.5, 5), 3.75);

    // 相同值测试
    EXPECT_DOUBLE_EQ(Midpoint(7, 7), 7.0);
    EXPECT_DOUBLE_EQ(Midpoint(3.14, 3.14), 3.14);

    // 大数值测试（避免溢出）
    EXPECT_DOUBLE_EQ(Midpoint(1e10, 3e10), 2e10);
    EXPECT_DOUBLE_EQ(Midpoint(-1e10, 1e10), 0.0);
}

TEST(UtilityFunctionsTest, MidpointEdgeCases)
{
    // 测试极值情况
    EXPECT_DOUBLE_EQ(Midpoint(std::numeric_limits<int>::min() + 1, std::numeric_limits<int>::max()), 0.0);

    // 测试浮点数精度
    float a   = 1.0f, b = 1.0f + std::numeric_limits<float>::epsilon();
    float mid = Midpoint(a, b);
    EXPECT_FLOAT_EQ(mid, a);
    EXPECT_FLOAT_EQ(mid, b);

    // 测试无穷大和 NaN 的处理
    EXPECT_TRUE(std::isfinite(Midpoint(1.0, 2.0)));

    // 测试非常接近的值
    double x        = 1.0, y = 1.0 + 1e-15;
    double midpoint = Midpoint(x, y);
    EXPECT_DOUBLE_EQ(midpoint, x);
    EXPECT_DOUBLE_EQ(midpoint, y);
}

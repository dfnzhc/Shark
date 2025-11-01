/**
 * @File TestMath.cu
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/31
 * @Brief This file is part of Shark.
 */

#include "TestMath.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "Shark/Math/Math.hpp"

using namespace SKT;

#define SKT_TEST(expr, expected) do { if (Abs((expr) - expected) > kEpsilonF) { errCnt += 1; printf("[Common Math] Error: " #expr " (%s:%d). \n", __FILE__, __LINE__);} } while(0)
#define SKT_TEST_NEAR(expr, expected, epsilon) do { if (Abs((expr) - expected) > epsilon) { errCnt += 1; printf("[Common Math] Error: " #expr " (%s:%d). \n", __FILE__, __LINE__);} } while(0)

// printf("[Common Math] Error: " #expr " (%s:%d). \n", __FILE__, __LINE__);

__global__ void TestCommonMathKernel(int* errCount)
{
    const int i    = 7;
    const float f  = 42.123f;
    const double d = 23.33;

    int errCnt = 0;

    SKT_TEST(Abs(-i), i);
    SKT_TEST(Abs(-f), f);
    SKT_TEST(Abs(-d), d);

    SKT_TEST(Min(i, f), i);
    SKT_TEST(Min(d, f), d);
    SKT_TEST(Min(d, i), i);

    SKT_TEST(Max(i, f), f);
    SKT_TEST(Max(d, f), f);
    SKT_TEST(Max(d, i), d);

    SKT_TEST(Min(i, f, d), i);
    SKT_TEST(Max(i, f, d), f);

    SKT_TEST(FMod(5.1f, 3.0f), 2.1f);
    SKT_TEST(Remainder(5.1f, 3.0f), -0.9f);
    SKT_TEST(FMA(2.0f, 3.0f, 1.0f), 7.0f);
    SKT_TEST(FDim(5.0f, 3.0f), 2.0f);
    SKT_TEST(FDim(3.0f, 5.0f), 0.0f);

    SKT_TEST(Sin(0.0f), 0.0f);
    SKT_TEST(Sin(kPi / 2.0f), 1.0f);
    SKT_TEST(Cos(0.0f), 1.0f);
    SKT_TEST(Cos(kPi), -1.0f);
    SKT_TEST(Tan(0.0f), 0.0f);

    SKT_TEST(ArcSin(0.0f), 0.0f);
    SKT_TEST(ArcSin(1.0f), kPi / 2.0f);
    SKT_TEST(ArcCos(1.0f), 0.0f);
    SKT_TEST(ArcTan(0.0f), 0.0f);
    SKT_TEST(ArcTan2(0.0f, 1.0f), 0.0f);

    SKT_TEST(Sinh(0.0f), 0.0f);
    SKT_TEST(Cosh(0.0f), 1.0f);
    SKT_TEST(Tanh(0.0f), 0.0f);
    SKT_TEST(ArcSinh(0.0f), 0.0f);
    SKT_TEST(ArcCosh(1.0f), 0.0f);
    SKT_TEST(ArcTanh(0.0f), 0.0f);

    SKT_TEST(Exp(0.0f), 1.0f);
    SKT_TEST(Exp(1.0f), kE);
    SKT_TEST(Exp2(0.0f), 1.0f);
    SKT_TEST(Exp2(1.0f), 2.0f);
    SKT_TEST(ExpM1(0.0f), 0.0f);

    SKT_TEST(Log(1.0f), 0.0f);
    SKT_TEST(Log(kE), 1.0f);
    SKT_TEST(Log10(1.0f), 0.0f);
    SKT_TEST(Log10(10.0f), 1.0f);
    SKT_TEST(Log2(1.0f), 0.0f);
    SKT_TEST(Log2(2.0f), 1.0f);
    SKT_TEST(Log1p(0.0f), 0.0f);

    SKT_TEST(Pow(2.0f, 3.0f), 8.0f);
    SKT_TEST(Pow(4.0f, 0.5f), 2.0f);
    SKT_TEST(Sqrt(4.0f), 2.0f);
    SKT_TEST(Sqrt(9.0f), 3.0f);
    SKT_TEST(Cbrt(8.0f), 2.0f);
    SKT_TEST(Cbrt(27.0f), 3.0f);
    SKT_TEST(Hypot(3.0f, 4.0f), 5.0f);

    SKT_TEST(Erf(0.0f), 0.0f);
    SKT_TEST(Erfc(0.0f), 1.0f);
    SKT_TEST(tGmma(1.0f), 1.0f);
    SKT_TEST(lGmma(1.0f), 0.0f);

    SKT_TEST(Ceil(2.3f), 3.0f);
    SKT_TEST(Ceil(-2.3f), -2.0f);
    SKT_TEST(Floor(2.7f), 2.0f);
    SKT_TEST(Floor(-2.3f), -3.0f);
    SKT_TEST(Trunc(2.7f), 2.0f);
    SKT_TEST(Trunc(-2.7f), -2.0f);
    SKT_TEST(Round(2.3f), 2.0f);
    SKT_TEST(Round(2.7f), 3.0f);
    SKT_TEST(NearbyInt(2.3f), 2.0f);

    int exp;
    float mantissa = Frexp(8.0f, &exp);
    SKT_TEST(mantissa, 0.5f);
    SKT_TEST(exp, 4);
    SKT_TEST(Ldexp(0.5f, 4), 8.0f);

    float intpart;
    float fracpart = Modf(3.14f, &intpart);
    SKT_TEST(intpart, 3.0f);
    SKT_TEST(fracpart, 0.14f);

    SKT_TEST(Logb(8.0f), 3);

    SKT_TEST(NextAfter(1.0f, 2.0f), 1.0000001f);
    SKT_TEST(CopySign(5.0f, -1.0f), -5.0f);
    SKT_TEST(CopySign(-5.0f, 1.0f), 5.0f);

    // 基础数学函数测试
    SKT_TEST(Mod(7, 3), 1);
    SKT_TEST(Mod(-7, 3), 2);
    SKT_TEST(Mod(7, -3), 1);
    SKT_TEST(InvHypot(3.0f, 4.0f), 0.2f);

    // 工具函数测试
    SKT_TEST(AlmostZero(1e-10f), true);
    SKT_TEST(AlmostZero(0.1f), false);
    SKT_TEST(Sinc(0.0f), 1.0f);
    SKT_TEST(Sinc(kPi), 0.0f);
    SKT_TEST(Sqr(3.0f), 9.0f);
    SKT_TEST(Sqr(-4.0f), 16.0f);

    // 角度转换测试
    SKT_TEST(Degree2Radian(180.0f), kPi);
    SKT_TEST(Degree2Radian(90.0f), kPi / 2.0f);
    SKT_TEST(Radian2Degree(kPi), 180.0f);
    SKT_TEST(Radian2Degree(kPi / 2.0f), 90.0f);

    // 插值和限制函数测试
    SKT_TEST(Lerp(0.0f, 10.0f, 0.5f), 5.0f);
    SKT_TEST(Lerp(2.0f, 8.0f, 0.25f), 3.5f);
    SKT_TEST(Clamp(5.0f, 0.0f, 10.0f), 5.0f);
    SKT_TEST(Clamp(-1.0f, 0.0f, 10.0f), 0.0f);
    SKT_TEST(Clamp(15.0f, 0.0f, 10.0f), 10.0f);
    SKT_TEST(Clamp01(0.5f), 0.5f);
    SKT_TEST(Clamp01(-0.1f), 0.0f);
    SKT_TEST(Clamp01(1.5f), 1.0f);

    // 高级数学函数测试
    SKT_TEST(Gaussian(0.0f, 0.0f, 1.0f), 0.39894228f); // 1/sqrt(2*pi)
    SKT_TEST(FastExp(0.0f), 1.0f);
    SKT_TEST(FastExp(1.0f), kE);
    SKT_TEST(FastSqrt(4.0f), 2.0f);
    SKT_TEST(FastSqrt(9.0f), 3.0f);
    SKT_TEST(FastCbrt(8.0f), 2.0f);
    SKT_TEST(FastCbrt(27.0f), 3.0f);
    SKT_TEST_NEAR(FastInvSqrt(4.0f), 0.5f, 1e-5);
    SKT_TEST_NEAR(FastInvSqrt(1.0f), 1.0f, 1e-5);

    // 多项式求值测试
    SKT_TEST(EvaluatePolynomial(2.0f, 1.0f), 1.0f);              // P(x) = 1
    SKT_TEST(EvaluatePolynomial(2.0f, 1.0f, 2.0f), 5.0f);        // P(x) = 1 + 2x = 1 + 2*2 = 5
    SKT_TEST(EvaluatePolynomial(3.0f, 1.0f, 2.0f, 1.0f), 16.0f); // P(x) = 1 + 2x + x^2 = 1 + 6 + 9 = 16

    // 二次方程求解测试
    float t0, t1;
    bool hasRoots = Quadratic(1.0f, -5.0f, 6.0f, &t0, &t1); // x^2 - 5x + 6 = 0, roots: 2, 3
    SKT_TEST(hasRoots, true);
    SKT_TEST(t0, 2.0f);
    SKT_TEST(t1, 3.0f);

    // 高斯积分测试（简单情况）
    float gaussIntegral = GaussianIntegral(-1.0f, 1.0f, 0.0f, 1.0f);
    SKT_TEST(gaussIntegral, 0.6826895f); // 约68.27%的概率在1个标准差内

    // 数论函数测试
    SKT_TEST(GCD(12, 8), 4);
    SKT_TEST(GCD(8, 12), 4);
    SKT_TEST(GCD(17, 13), 1); // 互质数
    SKT_TEST(GCD(100, 25), 25);
    SKT_TEST(GCD(7, 7), 7); // 相同数字

    SKT_TEST(LCM(4, 6), 12);
    SKT_TEST(LCM(6, 4), 12);
    SKT_TEST(LCM(7, 5), 35); // 互质数
    SKT_TEST(LCM(12, 18), 36);
    SKT_TEST(LCM(8, 8), 8); // 相同数字

    // 可变参数版本测试
    SKT_TEST(GCD(12, 8, 4), 4);
    SKT_TEST(GCD(60, 48, 36), 12);
    SKT_TEST(LCM(4, 6, 8), 24);
    SKT_TEST(LCM(2, 3, 4), 12);

    // GCD 和 LCM 关系验证: GCD(a,b) * LCM(a,b) = a * b
    SKT_TEST(GCD(12, 8) * LCM(12, 8), 12 * 8);
    SKT_TEST(GCD(15, 25) * LCM(15, 25), 15 * 25);

    // 中点函数测试
    SKT_TEST(Midpoint(0, 10), 5.0f);
    SKT_TEST(Midpoint(10, 0), 5.0f);
    SKT_TEST(Midpoint(-5, 5), 0.0f);
    SKT_TEST(Midpoint(3, 7), 5.0f);
    SKT_TEST(Midpoint(1.0f, 3.0f), 2.0f);
    SKT_TEST(Midpoint(-2.5f, 2.5f), 0.0f);
    SKT_TEST(Midpoint(0.0f, 1.0f), 0.5f);
    SKT_TEST(Midpoint(7, 7), 7.0f);
    SKT_TEST(Midpoint(3.14f, 3.14f), 3.14f);

    *errCount = errCnt;
}

bool TestCommonMath()
{
    thrust::host_vector<int> h_vec(1, 0);
    thrust::device_vector<int> d_vec = h_vec;

    TestCommonMathKernel<<<1, 1>>>(thrust::raw_pointer_cast(d_vec.data()));
    cudaDeviceSynchronize();

    h_vec = d_vec;

    return h_vec[0] == 0;
}

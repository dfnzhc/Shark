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
    
    SKT_TEST(Scalbn(1.0f, 3), 8.0f);
    SKT_TEST(Logb(8.0f), 3);
    
    SKT_TEST(NextAfter(1.0f, 2.0f), 1.0000001f);
    SKT_TEST(CopySign(5.0f, -1.0f), -5.0f);
    SKT_TEST(CopySign(-5.0f, 1.0f), 5.0f);
    

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

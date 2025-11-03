/**
 * @File TestVector.cu
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/3
 * @Brief This file is part of Shark.
 */


#include <cassert>
#include <cstdio>

#include "helper_cuda.h"
#include "helper_functions.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "Shark/Math/Math.hpp"

using namespace SKT;

#define SKT_TEST(expr, expected) do { if (Abs((expr) - (expected)) > kEpsilonF) { errCnt += 1; printf("[Vector Math] Error: " #expr " (%s:%d). \n", __FILE__, __LINE__);} } while(0)
#define SKT_TEST_NEAR(expr, expected, epsilon) do { if (Abs((expr) - (expected)) > epsilon) { errCnt += 1; printf("[Vector Math] Error: " #expr " (%s:%d). \n", __FILE__, __LINE__);} } while(0)

__global__ void TestVectorMathKernel(int* errCount)
{
    int errCnt = 0;

    // ==================== Vec1 测试 ====================

    // 构造函数测试
    {
        Vec1f v1;        // 默认构造
        Vec1f v2(3.14f); // 值构造
        Vec1f v3(v2);    // 拷贝构造
        Vec1i v4(42);    // 整数构造
        Vec1f v5(v4);    // 类型转换构造

        SKT_TEST(v1.x, 0);
        SKT_TEST(v2.x, 3.14f);
        SKT_TEST(v3.x, 3.14f);
        SKT_TEST(v4.x, 42.0f);
        SKT_TEST(v5.x, 42.0f);
    }

    // 访问操作测试
    {
        Vec1f v(5.0f);
        SKT_TEST(v[0], 5.0f);
        SKT_TEST(v.x, 5.0f);
        SKT_TEST(v.value(), 5.0f);
        SKT_TEST(static_cast<f32>(v), 5.0f);

        v.set(7.0f);
        SKT_TEST(v.x, 7.0f);
    }

    // 静态方法测试
    {
        Vec1f zero = Vec1f::Zero();
        Vec1f one  = Vec1f::One();
        Vec1f unit = Vec1f::Unit();

        SKT_TEST(zero.x, 0.0f);
        SKT_TEST(one.x, 1.0f);
        SKT_TEST(unit.x, 1.0f);
    }

    // ==================== Vec2 测试 ====================

    // 构造函数测试
    {
        Vec2f v1;             // 默认构造
        Vec2f v2(3.14f);      // 单值构造
        Vec2f v3(1.0f, 2.0f); // 双值构造
        Vec2f v4(v3);         // 拷贝构造
        Vec2i v5(10, 20);     // 整数构造
        Vec2f v6(v5);         // 类型转换构造

        SKT_TEST(v2.x, 3.14f);
        SKT_TEST(v2.y, 3.14f);
        SKT_TEST(v3.x, 1.0f);
        SKT_TEST(v3.y, 2.0f);
        SKT_TEST(v6.x, 10.0f);
        SKT_TEST(v6.y, 20.0f);
    }

    // 访问操作测试
    {
        Vec2f v(3.0f, 4.0f);
        SKT_TEST(v[0], 3.0f);
        SKT_TEST(v[1], 4.0f);
        SKT_TEST(v.x, 3.0f);
        SKT_TEST(v.y, 4.0f);

        v.set(5.0f, 6.0f);
        SKT_TEST(v.x, 5.0f);
        SKT_TEST(v.y, 6.0f);
    }

    // Swizzle 操作测试
    {
        Vec2f v(1.0f, 2.0f);
        Vec2f xx = v.xx();
        Vec2f xy = v.xy();
        Vec2f yx = v.yx();
        Vec2f yy = v.yy();

        SKT_TEST(xx.x, 1.0f);
        SKT_TEST(xx.y, 1.0f);
        SKT_TEST(xy.x, 1.0f);
        SKT_TEST(xy.y, 2.0f);
        SKT_TEST(yx.x, 2.0f);
        SKT_TEST(yx.y, 1.0f);
        SKT_TEST(yy.x, 2.0f);
        SKT_TEST(yy.y, 2.0f);
    }

    // 静态方法测试
    {
        Vec2f zero  = Vec2f::Zero();
        Vec2f one   = Vec2f::One();
        Vec2f unitX = Vec2f::UnitX();
        Vec2f unitY = Vec2f::UnitY();

        SKT_TEST(zero.x, 0.0f);
        SKT_TEST(zero.y, 0.0f);
        SKT_TEST(one.x, 1.0f);
        SKT_TEST(one.y, 1.0f);
        SKT_TEST(unitX.x, 1.0f);
        SKT_TEST(unitX.y, 0.0f);
        SKT_TEST(unitY.x, 0.0f);
        SKT_TEST(unitY.y, 1.0f);
    }

    // ==================== Vec3 测试 ====================

    // 构造函数测试
    {
        Vec3f v1;                          // 默认构造
        Vec3f v2(3.14f);                   // 单值构造
        Vec3f v3(1.0f, 2.0f, 3.0f);        // 三值构造
        Vec3f v4(Vec2f(1.0f, 2.0f), 3.0f); // Vec2 + 标量构造
        Vec3f v5(1.0f, Vec2f(2.0f, 3.0f)); // 标量 + Vec2 构造

        SKT_TEST(v2.x, 3.14f);
        SKT_TEST(v2.y, 3.14f);
        SKT_TEST(v2.z, 3.14f);
        SKT_TEST(v3.x, 1.0f);
        SKT_TEST(v3.y, 2.0f);
        SKT_TEST(v3.z, 3.0f);
        SKT_TEST(v4.x, 1.0f);
        SKT_TEST(v4.y, 2.0f);
        SKT_TEST(v4.z, 3.0f);
        SKT_TEST(v5.x, 1.0f);
        SKT_TEST(v5.y, 2.0f);
        SKT_TEST(v5.z, 3.0f);
    }

    // 访问操作测试
    {
        Vec3f v(1.0f, 2.0f, 3.0f);
        SKT_TEST(v[0], 1.0f);
        SKT_TEST(v[1], 2.0f);
        SKT_TEST(v[2], 3.0f);

        v.set(4.0f, 5.0f, 6.0f);
        SKT_TEST(v.x, 4.0f);
        SKT_TEST(v.y, 5.0f);
        SKT_TEST(v.z, 6.0f);
    }

    // 静态方法测试
    {
        Vec3f zero  = Vec3f::Zero();
        Vec3f one   = Vec3f::One();
        Vec3f unitX = Vec3f::UnitX();
        Vec3f unitY = Vec3f::UnitY();
        Vec3f unitZ = Vec3f::UnitZ();

        SKT_TEST(zero.x, 0.0f);
        SKT_TEST(zero.y, 0.0f);
        SKT_TEST(zero.z, 0.0f);
        SKT_TEST(one.x, 1.0f);
        SKT_TEST(one.y, 1.0f);
        SKT_TEST(one.z, 1.0f);
        SKT_TEST(unitX.x, 1.0f);
        SKT_TEST(unitX.y, 0.0f);
        SKT_TEST(unitX.z, 0.0f);
        SKT_TEST(unitY.x, 0.0f);
        SKT_TEST(unitY.y, 1.0f);
        SKT_TEST(unitY.z, 0.0f);
        SKT_TEST(unitZ.x, 0.0f);
        SKT_TEST(unitZ.y, 0.0f);
        SKT_TEST(unitZ.z, 1.0f);
    }

    // ==================== Vec4 测试 ====================

    // 构造函数测试
    {
        Vec4f v1;                                       // 默认构造
        Vec4f v2(3.14f);                                // 单值构造
        Vec4f v3(1.0f, 2.0f, 3.0f, 4.0f);               // 四值构造
        Vec4f v4(Vec2f(1.0f, 2.0f), 3.0f, 4.0f);        // Vec2 + 两个标量
        Vec4f v5(1.0f, Vec2f(2.0f, 3.0f), 4.0f);        // 标量 + Vec2 + 标量
        Vec4f v6(1.0f, 2.0f, Vec2f(3.0f, 4.0f));        // 两个标量 + Vec2
        Vec4f v7(Vec3f(1.0f, 2.0f, 3.0f), 4.0f);        // Vec3 + 标量
        Vec4f v8(1.0f, Vec3f(2.0f, 3.0f, 4.0f));        // 标量 + Vec3
        Vec4f v9(Vec2f(1.0f, 2.0f), Vec2f(3.0f, 4.0f)); // Vec2 + Vec2

        SKT_TEST(v2.x, 3.14f);
        SKT_TEST(v2.y, 3.14f);
        SKT_TEST(v2.z, 3.14f);
        SKT_TEST(v2.w, 3.14f);
        SKT_TEST(v3.x, 1.0f);
        SKT_TEST(v3.y, 2.0f);
        SKT_TEST(v3.z, 3.0f);
        SKT_TEST(v3.w, 4.0f);
        SKT_TEST(v9.x, 1.0f);
        SKT_TEST(v9.y, 2.0f);
        SKT_TEST(v9.z, 3.0f);
        SKT_TEST(v9.w, 4.0f);
    }

    // 访问操作测试
    {
        Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
        SKT_TEST(v[0], 1.0f);
        SKT_TEST(v[1], 2.0f);
        SKT_TEST(v[2], 3.0f);
        SKT_TEST(v[3], 4.0f);

        v.set(5.0f, 6.0f, 7.0f, 8.0f);
        SKT_TEST(v.x, 5.0f);
        SKT_TEST(v.y, 6.0f);
        SKT_TEST(v.z, 7.0f);
        SKT_TEST(v.w, 8.0f);
    }

    // 静态方法测试
    {
        Vec4f zero  = Vec4f::Zero();
        Vec4f one   = Vec4f::One();
        Vec4f unitX = Vec4f::UnitX();
        Vec4f unitY = Vec4f::UnitY();
        Vec4f unitZ = Vec4f::UnitZ();
        Vec4f unitW = Vec4f::UnitW();

        SKT_TEST(zero.x, 0.0f);
        SKT_TEST(zero.y, 0.0f);
        SKT_TEST(zero.z, 0.0f);
        SKT_TEST(zero.w, 0.0f);
        SKT_TEST(one.x, 1.0f);
        SKT_TEST(one.y, 1.0f);
        SKT_TEST(one.z, 1.0f);
        SKT_TEST(one.w, 1.0f);
        SKT_TEST(unitX.x, 1.0f);
        SKT_TEST(unitX.y, 0.0f);
        SKT_TEST(unitX.z, 0.0f);
        SKT_TEST(unitX.w, 0.0f);
        SKT_TEST(unitY.x, 0.0f);
        SKT_TEST(unitY.y, 1.0f);
        SKT_TEST(unitY.z, 0.0f);
        SKT_TEST(unitY.w, 0.0f);
        SKT_TEST(unitZ.x, 0.0f);
        SKT_TEST(unitZ.y, 0.0f);
        SKT_TEST(unitZ.z, 1.0f);
        SKT_TEST(unitZ.w, 0.0f);
        SKT_TEST(unitW.x, 0.0f);
        SKT_TEST(unitW.y, 0.0f);
        SKT_TEST(unitW.z, 0.0f);
        SKT_TEST(unitW.w, 1.0f);
    }

    // ==================== 算术运算测试 ====================

    // 向量加法测试
    {
        Vec2f a(1.0f, 2.0f);
        Vec2f b(3.0f, 4.0f);
        Vec2f c = a + b;
        SKT_TEST(c.x, 4.0f);
        SKT_TEST(c.y, 6.0f);

        // 向量与标量加法
        Vec2f d = a + 5.0f;
        SKT_TEST(d.x, 6.0f);
        SKT_TEST(d.y, 7.0f);

        // 标量与向量加法
        Vec2f e = 5.0f + a;
        SKT_TEST(e.x, 6.0f);
        SKT_TEST(e.y, 7.0f);
    }

    // 向量减法测试
    {
        Vec2f a(5.0f, 7.0f);
        Vec2f b(2.0f, 3.0f);
        Vec2f c = a - b;
        SKT_TEST(c.x, 3.0f);
        SKT_TEST(c.y, 4.0f);

        // 向量与标量减法
        Vec2f d = a - 1.0f;
        SKT_TEST(d.x, 4.0f);
        SKT_TEST(d.y, 6.0f);

        // 标量与向量减法
        Vec2f e = 10.0f - a;
        SKT_TEST(e.x, 5.0f);
        SKT_TEST(e.y, 3.0f);
    }

    // 向量乘法测试
    {
        Vec2f a(2.0f, 3.0f);
        Vec2f b(4.0f, 5.0f);
        Vec2f c = a * b;
        SKT_TEST(c.x, 8.0f);
        SKT_TEST(c.y, 15.0f);

        // 向量与标量乘法
        Vec2f d = a * 2.0f;
        SKT_TEST(d.x, 4.0f);
        SKT_TEST(d.y, 6.0f);

        // 标量与向量乘法
        Vec2f e = 3.0f * a;
        SKT_TEST(e.x, 6.0f);
        SKT_TEST(e.y, 9.0f);
    }

    // 向量除法测试
    {
        Vec2f a(8.0f, 12.0f);
        Vec2f b(2.0f, 3.0f);
        Vec2f c = a / b;
        SKT_TEST(c.x, 4.0f);
        SKT_TEST(c.y, 4.0f);

        // 向量与标量除法
        Vec2f d = a / 2.0f;
        SKT_TEST(d.x, 4.0f);
        SKT_TEST(d.y, 6.0f);

        // 标量与向量除法
        Vec2f e = 24.0f / a;
        SKT_TEST(e.x, 3.0f);
        SKT_TEST(e.y, 2.0f);
    }

    // 一元运算符测试
    {
        Vec2f a(3.0f, -4.0f);
        Vec2f pos = +a;
        Vec2f neg = -a;

        SKT_TEST(pos.x, 3.0f);
        SKT_TEST(pos.y, -4.0f);
        SKT_TEST(neg.x, -3.0f);
        SKT_TEST(neg.y, 4.0f);
    }

    // 复合赋值运算符测试
    {
        Vec2f a(2.0f, 3.0f);
        Vec2f b(1.0f, 2.0f);

        a += b;
        SKT_TEST(a.x, 3.0f);
        SKT_TEST(a.y, 5.0f);

        a -= Vec2f(1.0f, 1.0f);
        SKT_TEST(a.x, 2.0f);
        SKT_TEST(a.y, 4.0f);

        a *= 2.0f;
        SKT_TEST(a.x, 4.0f);
        SKT_TEST(a.y, 8.0f);

        a /= 2.0f;
        SKT_TEST(a.x, 2.0f);
        SKT_TEST(a.y, 4.0f);
    }

    // ==================== 比较运算测试 ====================

    // 相等性测试
    {
        Vec2f a(1.0f, 2.0f);
        Vec2f b(1.0f, 2.0f);
        Vec2f c(1.0f, 3.0f);

        if (!(a == b))
            errCnt++;
        if (a == c)
            errCnt++;
        if (a != b)
            errCnt++;
        if (!(a != c))
            errCnt++;
    }

    // 元素级比较测试
    {
        Vec2f a(1.0f, 4.0f);
        Vec2f b(2.0f, 3.0f);

        Vec2<bool> lt = LessThan(a, b);
        Vec2<bool> gt = GreaterThan(a, b);
        Vec2<bool> eq = Equal(a, Vec2f(1.0f, 4.0f));

        if (!lt.x || lt.y)
            errCnt++; // 1 < 2 (true), 4 < 3 (false)
        if (gt.x || !gt.y)
            errCnt++; // 1 > 2 (false), 4 > 3 (true)
        if (!eq.x || !eq.y)
            errCnt++; // 1 == 1 (true), 4 == 4 (true)
    }

    // 逻辑运算测试
    {
        Vec2<bool> a(true, false);
        Vec2<bool> b(false, true);

        Vec2<bool> and_result = a && b;
        Vec2<bool> or_result  = a || b;
        Vec2<bool> not_result = !a;

        if (and_result.x || and_result.y)
            errCnt++; // false && false, true && false
        if (!or_result.x || !or_result.y)
            errCnt++; // true || false, false || true
        if (not_result.x || !not_result.y)
            errCnt++; // !true, !false

        // All, Any, None, Count 测试
        if (!All(Vec2<bool>(true, true)))
            errCnt++;
        if (All(Vec2<bool>(true, false)))
            errCnt++;
        if (!Any(Vec2<bool>(true, false)))
            errCnt++;
        if (Any(Vec2<bool>(false, false)))
            errCnt++;
        if (!None(Vec2<bool>(false, false)))
            errCnt++;
        if (None(Vec2<bool>(true, false)))
            errCnt++;
        if (Count(Vec2<bool>(true, true)) != 2)
            errCnt++;
        if (Count(Vec2<bool>(true, false)) != 1)
            errCnt++;
    }

    // ==================== 位运算测试 ====================

    // 整数向量位运算测试
    {
        Vec2i a(0b1010, 0b1100); // 10, 12
        Vec2i b(0b1100, 0b1010); // 12, 10

        Vec2i and_result = a & b;
        Vec2i or_result  = a | b;
        Vec2i xor_result = a ^ b;
        Vec2i not_result = ~a;

        SKT_TEST(and_result.x, 0b1000); // 8
        SKT_TEST(and_result.y, 0b1000); // 8
        SKT_TEST(or_result.x, 0b1110);  // 14
        SKT_TEST(or_result.y, 0b1110);  // 14
        SKT_TEST(xor_result.x, 0b0110); // 6
        SKT_TEST(xor_result.y, 0b0110); // 6

        // 位移运算测试
        Vec2i left_shift  = a << 1;
        Vec2i right_shift = a >> 1;

        SKT_TEST(left_shift.x, 20); // 10 << 1
        SKT_TEST(left_shift.y, 24); // 12 << 1
        SKT_TEST(right_shift.x, 5); // 10 >> 1
        SKT_TEST(right_shift.y, 6); // 12 >> 1

        // 向量与标量位运算
        Vec2i scalar_and = a & 0b1111;
        SKT_TEST(scalar_and.x, 10);
        SKT_TEST(scalar_and.y, 12);

        // 复合位运算赋值
        Vec2i c = a;
        c &= b;
        SKT_TEST(c.x, 0b1000);
        SKT_TEST(c.y, 0b1000);

        c = a;
        c |= b;
        SKT_TEST(c.x, 0b1110);
        SKT_TEST(c.y, 0b1110);

        c = a;
        c ^= b;
        SKT_TEST(c.x, 0b0110);
        SKT_TEST(c.y, 0b0110);

        c = a;
        c <<= 1;
        SKT_TEST(c.x, 20);
        SKT_TEST(c.y, 24);

        c = a;
        c >>= 1;
        SKT_TEST(c.x, 5);
        SKT_TEST(c.y, 6);
    }

    // ==================== 模运算测试 ====================

    // 整数向量模运算测试
    {
        Vec2i a(10, 15);
        Vec2i b(3, 4);

        Vec2i mod_result = a % b;
        SKT_TEST(mod_result.x, 1); // 10 % 3
        SKT_TEST(mod_result.y, 3); // 15 % 4

        // 向量与标量模运算
        Vec2i scalar_mod = a % 7;
        SKT_TEST(scalar_mod.x, 3); // 10 % 7
        SKT_TEST(scalar_mod.y, 1); // 15 % 7

        // 复合模运算赋值
        Vec2i c = a;
        c %= b;
        SKT_TEST(c.x, 1);
        SKT_TEST(c.y, 3);
    }

    // ==================== Vec3 数学运算测试 ====================

    // Vec3 算术运算测试
    {
        Vec3f a(1.0f, 2.0f, 3.0f);
        Vec3f b(4.0f, 5.0f, 6.0f);

        // 加法
        Vec3f add_result = a + b;
        SKT_TEST(add_result.x, 5.0f);
        SKT_TEST(add_result.y, 7.0f);
        SKT_TEST(add_result.z, 9.0f);

        // 减法
        Vec3f sub_result = b - a;
        SKT_TEST(sub_result.x, 3.0f);
        SKT_TEST(sub_result.y, 3.0f);
        SKT_TEST(sub_result.z, 3.0f);

        // 乘法
        Vec3f mul_result = a * Vec3f(2.0f, 3.0f, 4.0f);
        SKT_TEST(mul_result.x, 2.0f);
        SKT_TEST(mul_result.y, 6.0f);
        SKT_TEST(mul_result.z, 12.0f);

        // 标量乘法
        Vec3f scalar_mul = a * 2.0f;
        SKT_TEST(scalar_mul.x, 2.0f);
        SKT_TEST(scalar_mul.y, 4.0f);
        SKT_TEST(scalar_mul.z, 6.0f);
    }

    // ==================== Vec4 数学运算测试 ====================

    // Vec4 算术运算测试
    {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(5.0f, 6.0f, 7.0f, 8.0f);

        // 加法
        Vec4f add_result = a + b;
        SKT_TEST(add_result.x, 6.0f);
        SKT_TEST(add_result.y, 8.0f);
        SKT_TEST(add_result.z, 10.0f);
        SKT_TEST(add_result.w, 12.0f);

        // 减法
        Vec4f sub_result = b - a;
        SKT_TEST(sub_result.x, 4.0f);
        SKT_TEST(sub_result.y, 4.0f);
        SKT_TEST(sub_result.z, 4.0f);
        SKT_TEST(sub_result.w, 4.0f);

        // 标量除法
        Vec4f div_result = b / 2.0f;
        SKT_TEST(div_result.x, 2.5f);
        SKT_TEST(div_result.y, 3.0f);
        SKT_TEST(div_result.z, 3.5f);
        SKT_TEST(div_result.w, 4.0f);
    }

    // ==================== 混合类型运算测试 ====================

    // 不同类型向量运算
    {
        Vec2i int_vec(3, 4);
        Vec2f float_vec(1.5f, 2.5f);

        // 整数向量与浮点向量运算
        Vec2f mixed_result = int_vec + float_vec;
        SKT_TEST(mixed_result.x, 4.5f);
        SKT_TEST(mixed_result.y, 6.5f);

        // 整数向量与浮点标量运算
        Vec2f scalar_mixed = int_vec * 1.5f;
        SKT_TEST(scalar_mixed.x, 4.5f);
        SKT_TEST(scalar_mixed.y, 6.0f);
    }

    // ==================== 边界情况测试 ====================

    // 零向量测试
    {
        Vec2f zero = Vec2f::Zero();
        Vec2f a(3.0f, 4.0f);

        Vec2f zero_add = a + zero;
        SKT_TEST(zero_add.x, 3.0f);
        SKT_TEST(zero_add.y, 4.0f);

        Vec2f zero_mul = a * 0.0f;
        SKT_TEST(zero_mul.x, 0.0f);
        SKT_TEST(zero_mul.y, 0.0f);
    }

    // 单位向量测试
    {
        Vec3f unitX = Vec3f::UnitX();
        Vec3f unitY = Vec3f::UnitY();
        Vec3f unitZ = Vec3f::UnitZ();

        // 单位向量正交性测试（元素级乘法应该为零）
        Vec3f xy_mul = unitX * unitY;
        Vec3f xz_mul = unitX * unitZ;
        Vec3f yz_mul = unitY * unitZ;

        SKT_TEST(xy_mul.x, 0.0f);
        SKT_TEST(xy_mul.y, 0.0f);
        SKT_TEST(xy_mul.z, 0.0f);
        SKT_TEST(xz_mul.x, 0.0f);
        SKT_TEST(xz_mul.y, 0.0f);
        SKT_TEST(xz_mul.z, 0.0f);
        SKT_TEST(yz_mul.x, 0.0f);
        SKT_TEST(yz_mul.y, 0.0f);
        SKT_TEST(yz_mul.z, 0.0f);
    }

    // 大数值测试
    {
        Vec2f large_vec(1e6f, 2e6f);
        Vec2f small_vec(1e-6f, 2e-6f);

        Vec2f large_add = large_vec + Vec2f(1.0f, 1.0f);
        SKT_TEST_NEAR(large_add.x, 1e6f + 1.0f, 1e-3f);
        SKT_TEST_NEAR(large_add.y, 2e6f + 1.0f, 1e-3f);

        Vec2f small_mul = small_vec * 1e6f;
        SKT_TEST_NEAR(small_mul.x, 1.0f, 1e-6f);
        SKT_TEST_NEAR(small_mul.y, 2.0f, 1e-6f);
    }

    // 负数测试
    {
        Vec2f neg_vec(-3.0f, -4.0f);
        Vec2f pos_vec(3.0f, 4.0f);

        Vec2f neg_add = neg_vec + pos_vec;
        SKT_TEST(neg_add.x, 0.0f);
        SKT_TEST(neg_add.y, 0.0f);

        Vec2f abs_test = -neg_vec;
        SKT_TEST(abs_test.x, 3.0f);
        SKT_TEST(abs_test.y, 4.0f);
    }

    // ==================== 复杂运算链测试 ====================

    // 多步运算测试
    {
        Vec2f a(1.0f, 2.0f);
        Vec2f b(3.0f, 4.0f);
        Vec2f c(5.0f, 6.0f);

        // 复杂表达式: (a + b) * c - a / 2
        Vec2f complex_result = (a + b) * c - a / 2.0f;

        // 手动计算验证
        // (1+3, 2+4) * (5, 6) - (1/2, 2/2) = (4, 6) * (5, 6) - (0.5, 1) = (20, 36) - (0.5, 1) = (19.5, 35)
        SKT_TEST(complex_result.x, 19.5f);
        SKT_TEST(complex_result.y, 35.0f);
    }

    // 链式赋值测试
    {
        Vec2f a(2.0f, 3.0f);
        Vec2f b = a;

        b += Vec2f(1.0f, 1.0f);
        b *= 2.0f;
        b -= a;

        // 计算过程: (2,3) -> (3,4) -> (6,8) -> (4,5)
        SKT_TEST(b.x, 4.0f);
        SKT_TEST(b.y, 5.0f);
    }

    // ==================== 数据类型测试 ====================

    // 不同精度浮点数测试
    {
        Vec2d double_vec(1.123456789, 2.987654321);
        Vec2f float_vec = Vec2f(double_vec); // 类型转换

        SKT_TEST_NEAR(float_vec.x, 1.123456789f, 1e-6f);
        SKT_TEST_NEAR(float_vec.y, 2.987654321f, 1e-6f);
    }

    // 整数类型测试
    {
        Vec2i int_vec(100, 200);
        Vec2u uint_vec(300u, 400u);

        // 有符号和无符号整数运算
        Vec2i mixed_int = int_vec + Vec2i(uint_vec);
        SKT_TEST(mixed_int.x, 400);
        SKT_TEST(mixed_int.y, 600);
    }

    // ==================== 高级 Swizzle 操作测试 ====================

    // Vec2 Swizzle 全面测试
    {
        Vec2f v(1.0f, 2.0f);

        // 2D swizzle 测试
        Vec2f xx = v.xx();
        SKT_TEST(xx.x, 1.0f);
        SKT_TEST(xx.y, 1.0f);
        Vec2f xy = v.xy();
        SKT_TEST(xy.x, 1.0f);
        SKT_TEST(xy.y, 2.0f);
        Vec2f yx = v.yx();
        SKT_TEST(yx.x, 2.0f);
        SKT_TEST(yx.y, 1.0f);
        Vec2f yy = v.yy();
        SKT_TEST(yy.x, 2.0f);
        SKT_TEST(yy.y, 2.0f);

        // 3D swizzle 测试
        Vec3f xxx = v.xxx();
        SKT_TEST(xxx.x, 1.0f);
        SKT_TEST(xxx.y, 1.0f);
        SKT_TEST(xxx.z, 1.0f);
        Vec3f xxy = v.xxy();
        SKT_TEST(xxy.x, 1.0f);
        SKT_TEST(xxy.y, 1.0f);
        SKT_TEST(xxy.z, 2.0f);
        Vec3f xyx = v.xyx();
        SKT_TEST(xyx.x, 1.0f);
        SKT_TEST(xyx.y, 2.0f);
        SKT_TEST(xyx.z, 1.0f);
        Vec3f xyy = v.xyy();
        SKT_TEST(xyy.x, 1.0f);
        SKT_TEST(xyy.y, 2.0f);
        SKT_TEST(xyy.z, 2.0f);
        Vec3f yxx = v.yxx();
        SKT_TEST(yxx.x, 2.0f);
        SKT_TEST(yxx.y, 1.0f);
        SKT_TEST(yxx.z, 1.0f);
        Vec3f yxy = v.yxy();
        SKT_TEST(yxy.x, 2.0f);
        SKT_TEST(yxy.y, 1.0f);
        SKT_TEST(yxy.z, 2.0f);
        Vec3f yyx = v.yyx();
        SKT_TEST(yyx.x, 2.0f);
        SKT_TEST(yyx.y, 2.0f);
        SKT_TEST(yyx.z, 1.0f);
        Vec3f yyy = v.yyy();
        SKT_TEST(yyy.x, 2.0f);
        SKT_TEST(yyy.y, 2.0f);
        SKT_TEST(yyy.z, 2.0f);

        // 4D swizzle 测试
        Vec4f xxxx = v.xxxx();
        SKT_TEST(xxxx.x, 1.0f);
        SKT_TEST(xxxx.y, 1.0f);
        SKT_TEST(xxxx.z, 1.0f);
        SKT_TEST(xxxx.w, 1.0f);
        Vec4f xyxy = v.xyxy();
        SKT_TEST(xyxy.x, 1.0f);
        SKT_TEST(xyxy.y, 2.0f);
        SKT_TEST(xyxy.z, 1.0f);
        SKT_TEST(xyxy.w, 2.0f);
        Vec4f yyyy = v.yyyy();
        SKT_TEST(yyyy.x, 2.0f);
        SKT_TEST(yyyy.y, 2.0f);
        SKT_TEST(yyyy.z, 2.0f);
        SKT_TEST(yyyy.w, 2.0f);
    }

    // Vec3 Swizzle 测试
    {
        Vec3f v(1.0f, 2.0f, 3.0f);

        // 2D swizzle 从 3D
        Vec2f xy = v.xy();
        SKT_TEST(xy.x, 1.0f);
        SKT_TEST(xy.y, 2.0f);
        Vec2f xz = v.xz();
        SKT_TEST(xz.x, 1.0f);
        SKT_TEST(xz.y, 3.0f);
        Vec2f yz = v.yz();
        SKT_TEST(yz.x, 2.0f);
        SKT_TEST(yz.y, 3.0f);
        Vec2f zx = v.zx();
        SKT_TEST(zx.x, 3.0f);
        SKT_TEST(zx.y, 1.0f);
        Vec2f zy = v.zy();
        SKT_TEST(zy.x, 3.0f);
        SKT_TEST(zy.y, 2.0f);

        // 3D swizzle 重排
        Vec3f xyz = v.xyz();
        SKT_TEST(xyz.x, 1.0f);
        SKT_TEST(xyz.y, 2.0f);
        SKT_TEST(xyz.z, 3.0f);
        Vec3f zyx = v.zyx();
        SKT_TEST(zyx.x, 3.0f);
        SKT_TEST(zyx.y, 2.0f);
        SKT_TEST(zyx.z, 1.0f);
        Vec3f xzy = v.xzy();
        SKT_TEST(xzy.x, 1.0f);
        SKT_TEST(xzy.y, 3.0f);
        SKT_TEST(xzy.z, 2.0f);
    }

    // ==================== 向量构造组合测试 ====================

    // Vec3 从不同组合构造
    {
        Vec2f v2(1.0f, 2.0f);
        f32 scalar = 3.0f;

        // Vec2 + 标量构造 Vec3
        Vec3f v3a(v2, scalar);
        SKT_TEST(v3a.x, 1.0f);
        SKT_TEST(v3a.y, 2.0f);
        SKT_TEST(v3a.z, 3.0f);

        // 标量 + Vec2 构造 Vec3
        Vec3f v3b(scalar, v2);
        SKT_TEST(v3b.x, 3.0f);
        SKT_TEST(v3b.y, 1.0f);
        SKT_TEST(v3b.z, 2.0f);
    }

    // Vec4 从不同组合构造
    {
        Vec2f v2a(1.0f, 2.0f);
        Vec2f v2b(3.0f, 4.0f);
        Vec3f v3(1.0f, 2.0f, 3.0f);
        f32 scalar = 4.0f;

        // Vec2 + Vec2 构造 Vec4
        Vec4f v4a(v2a, v2b);
        SKT_TEST(v4a.x, 1.0f);
        SKT_TEST(v4a.y, 2.0f);
        SKT_TEST(v4a.z, 3.0f);
        SKT_TEST(v4a.w, 4.0f);

        // Vec3 + 标量构造 Vec4
        Vec4f v4b(v3, scalar);
        SKT_TEST(v4b.x, 1.0f);
        SKT_TEST(v4b.y, 2.0f);
        SKT_TEST(v4b.z, 3.0f);
        SKT_TEST(v4b.w, 4.0f);

        // 标量 + Vec3 构造 Vec4
        Vec4f v4c(scalar, v3);
        SKT_TEST(v4c.x, 4.0f);
        SKT_TEST(v4c.y, 1.0f);
        SKT_TEST(v4c.z, 2.0f);
        SKT_TEST(v4c.w, 3.0f);

        // Vec2 + 两个标量构造 Vec4
        Vec4f v4d(v2a, 3.0f, 4.0f);
        SKT_TEST(v4d.x, 1.0f);
        SKT_TEST(v4d.y, 2.0f);
        SKT_TEST(v4d.z, 3.0f);
        SKT_TEST(v4d.w, 4.0f);

        // 标量 + Vec2 + 标量构造 Vec4
        Vec4f v4e(1.0f, v2b, 5.0f);
        SKT_TEST(v4e.x, 1.0f);
        SKT_TEST(v4e.y, 3.0f);
        SKT_TEST(v4e.z, 4.0f);
        SKT_TEST(v4e.w, 5.0f);

        // 两个标量 + Vec2 构造 Vec4
        Vec4f v4f(1.0f, 2.0f, v2b);
        SKT_TEST(v4f.x, 1.0f);
        SKT_TEST(v4f.y, 2.0f);
        SKT_TEST(v4f.z, 3.0f);
        SKT_TEST(v4f.w, 4.0f);
    }

    // ==================== 向量元素访问高级测试 ====================

    // 数组式访问测试
    {
        Vec4f v(10.0f, 20.0f, 30.0f, 40.0f);

        // 读取测试
        for (int i = 0; i < 4; ++i)
        {
            SKT_TEST(v[i], (i + 1) * 10.0f);
        }

        // 修改测试（如果支持）
        Vec4f v_copy = v;
        // 注意：这里假设 operator[] 返回引用，如果不支持修改则跳过此测试
    }

    // 成员访问一致性测试
    {
        Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);

        // 验证不同访问方式的一致性
        SKT_TEST(v.x, v[0]);
        SKT_TEST(v.y, v[1]);
        SKT_TEST(v.z, v[2]);
        SKT_TEST(v.w, v[3]);
    }

    // ==================== 向量类型转换高级测试 ====================

    // 精度转换测试
    {
        // 高精度到低精度
        Vec2d high_precision(1.123456789012345, 2.987654321098765);
        Vec2f low_precision(high_precision);

        // 验证转换后的精度损失在可接受范围内
        SKT_TEST_NEAR(low_precision.x, 1.123456789012345f, 1e-6f);
        SKT_TEST_NEAR(low_precision.y, 2.987654321098765f, 1e-6f);

        // 低精度到高精度
        Vec2f original(3.14f, 2.71f);
        Vec2d extended(original);
        SKT_TEST_NEAR(extended.x, 3.14, 1e-6);
        SKT_TEST_NEAR(extended.y, 2.71, 1e-6);
    }

    // 整数类型转换测试
    {
        // 有符号到无符号
        Vec2i signed_vec(100, 200);
        Vec2u unsigned_vec(signed_vec);
        SKT_TEST(unsigned_vec.x, 100u);
        SKT_TEST(unsigned_vec.y, 200u);

        // 不同大小整数转换
        Vec2i int_vec(1000, 2000);
        Vec2<i16> short_vec(int_vec); // 可能会有截断
        SKT_TEST(short_vec.x, 1000);
        SKT_TEST(short_vec.y, 2000);
    }

    // ==================== 性能相关测试 ====================

    // 大量运算测试（验证没有明显的性能问题）
    {
        Vec2f accumulator    = Vec2f::Zero();
        const int iterations = 1000;

        for (int i = 0; i < iterations; ++i)
        {
            Vec2f temp(static_cast<f32>(i), static_cast<f32>(i * 2));
            accumulator += temp;
            accumulator *= 0.999f; // 防止数值溢出
        }

        // 验证计算结果在合理范围内
        if (accumulator.x < -1e6f || accumulator.x > 1e6f)
            errCnt++;
        if (accumulator.y < -1e6f || accumulator.y > 1e6f)
            errCnt++;
    }

    *errCount = errCnt;
}

bool TestVectorMath()
{
    thrust::host_vector<int> h_vec(1, 0);
    thrust::device_vector<int> d_vec = h_vec;

    TestVectorMathKernel<<<1, 1>>>(thrust::raw_pointer_cast(d_vec.data()));
    cudaDeviceSynchronize();

    h_vec = d_vec;

    return h_vec[0] == 0;
}

#undef SKT_TEST
#undef SKT_TEST_NEAR

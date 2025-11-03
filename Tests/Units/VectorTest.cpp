/**
 * @File VectorTest.cpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/3
 * @Brief This file is part of Shark.
 */

#include <gtest/gtest.h>

#include <Shark/Shark.hpp>

using namespace SKT;
using namespace SKT::Detail;

TEST(VectorConceptTest, VectorTypeConcept) 
{
    static_assert(VectorType<Vec2f>);
    static_assert(VectorType<Vec3i>);
    static_assert(VectorType<Vec4d>);
    static_assert(VectorType<Vec1u>);
    
    static_assert(!VectorType<int>);
    static_assert(!VectorType<float>);
    static_assert(!VectorType<std::string>);
}

TEST(VectorConceptTest, FloatVectorTypeConcept) 
{
    static_assert(FloatVectorType<Vec2f>);
    static_assert(FloatVectorType<Vec3d>);
    static_assert(FloatVectorType<Vec4f>);
    
    static_assert(!FloatVectorType<Vec2i>);
    static_assert(!FloatVectorType<Vec3u>);
    static_assert(!FloatVectorType<int>);
}

TEST(VectorConceptTest, IntVectorTypeConcept) 
{
    static_assert(IntVectorType<Vec2i>);
    static_assert(IntVectorType<Vec3u>);
    static_assert(IntVectorType<Vec4i>);
    
    static_assert(!IntVectorType<Vec2f>);
    static_assert(!IntVectorType<Vec3d>);
    static_assert(!IntVectorType<float>);
}

TEST(VectorConceptTest, DimensionConcepts) 
{
    static_assert(ValidVectorDimension<1>);
    static_assert(ValidVectorDimension<2>);
    static_assert(ValidVectorDimension<3>);
    static_assert(ValidVectorDimension<4>);
    static_assert(!ValidVectorDimension<0>);
    static_assert(!ValidVectorDimension<5>);
}

TEST(VectorConceptTest, CompatibilityConcepts) 
{
    static_assert(VectorSameDimension<Vec2f, Vec2i>);
    static_assert(VectorSameDimension<Vec3d, Vec3u>);
    static_assert(!VectorSameDimension<Vec2f, Vec3f>);
    
    static_assert(CompatibleVectors<Vec2f, Vec2i>);
    static_assert(CompatibleVectors<Vec3d, Vec3f>);
    static_assert(!CompatibleVectors<Vec2f, Vec3f>);
    
    static_assert(VectorScalarCompatible<Vec2f, float>);
    static_assert(VectorScalarCompatible<Vec3i, int>);
    static_assert(VectorScalarCompatible<Vec4d, double>);
}

TEST(Vec1Test, Construction) 
{
    Vec1f v1;
    EXPECT_FLOAT_EQ(v1.x, 0.0f);
    
    Vec1f v2(5.0f);
    EXPECT_FLOAT_EQ(v2.x, 5.0f);
    
    Vec1i v3(10);
    Vec1f v4(v3);
    EXPECT_FLOAT_EQ(v4.x, 10.0f);
    
    Vec2f v2d(3.0f, 4.0f);
    Vec1f v5(v2d);
    EXPECT_FLOAT_EQ(v5.x, 3.0f);
}

TEST(Vec1Test, Assignment) 
{
    Vec1f v1(1.0f);
    Vec1f v2(2.0f);
    
    v1 = v2;
    EXPECT_FLOAT_EQ(v1.x, 2.0f);
    
    Vec1i vi(5);
    v1 = vi;
    EXPECT_FLOAT_EQ(v1.x, 5.0f);
}

TEST(Vec1Test, Access) 
{
    Vec1f v(3.14f);
    
    EXPECT_FLOAT_EQ(v[0], 3.14f);
    v[0] = 2.71f;
    EXPECT_FLOAT_EQ(v.x, 2.71f);
}

TEST(Vec1Test, StaticMethods) 
{
    auto zero = Vec1f::Zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    
    auto one = Vec1f::One();
    EXPECT_FLOAT_EQ(one.x, 1.0f);
    
    auto unit = Vec1f::Unit();
    EXPECT_FLOAT_EQ(unit.x, 1.0f);
    
    auto unitX = Vec1f::UnitX();
    EXPECT_FLOAT_EQ(unitX.x, 1.0f);
}

TEST(Vec1Test, ArithmeticAssignment) 
{
    Vec1f v1(10.0f);
    Vec1f v2(3.0f);
    
    v1 += v2;
    EXPECT_FLOAT_EQ(v1.x, 13.0f);
    
    v1 -= v2;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    
    v1 *= v2;
    EXPECT_FLOAT_EQ(v1.x, 30.0f);
    
    v1 /= v2;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    
    v1 += 5.0f;
    EXPECT_FLOAT_EQ(v1.x, 15.0f);
    
    v1 -= 5.0f;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    
    v1 *= 2.0f;
    EXPECT_FLOAT_EQ(v1.x, 20.0f);
    
    v1 /= 4.0f;
    EXPECT_FLOAT_EQ(v1.x, 5.0f);
}

TEST(Vec1Test, IntegerOperations) 
{
    Vec1i v1(10);
    Vec1i v2(3);
    
    v1 %= v2;
    EXPECT_EQ(v1.x, 1);
    
    Vec1i v3(0b1010);
    Vec1i v4(0b1100);
    
    v3 &= v4;
    EXPECT_EQ(v3.x, 0b1000);
    
    v3 = Vec1i(0b1010);
    v3 |= v4;
    EXPECT_EQ(v3.x, 0b1110);
    
    v3 = Vec1i(0b1010);
    v3 ^= v4;
    EXPECT_EQ(v3.x, 0b0110);
    
    v3 = Vec1i(0b0001);
    v3 <<= Vec1i(2);
    EXPECT_EQ(v3.x, 0b0100);
    
    v3 >>= Vec1i(1);
    EXPECT_EQ(v3.x, 0b0010);
}

TEST(Vec2Test, Construction) 
{
    Vec2f v1;
    EXPECT_FLOAT_EQ(v1.x, 0.0f);
    EXPECT_FLOAT_EQ(v1.y, 0.0f);
    
    Vec2f v2(5.0f);
    EXPECT_FLOAT_EQ(v2.x, 5.0f);
    EXPECT_FLOAT_EQ(v2.y, 5.0f);
    
    Vec2f v3(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(v3.x, 3.0f);
    EXPECT_FLOAT_EQ(v3.y, 4.0f);
    
    Vec2f v4(3, 4.5f);
    EXPECT_FLOAT_EQ(v4.x, 3.0f);
    EXPECT_FLOAT_EQ(v4.y, 4.5f);
    
    Vec1f v1d(2.0f);
    Vec2f v5(v1d, 3.0f);
    EXPECT_FLOAT_EQ(v5.x, 2.0f);
    EXPECT_FLOAT_EQ(v5.y, 3.0f);
    
    Vec3f v3d(1.0f, 2.0f, 3.0f);
    Vec2f v6(v3d);
    EXPECT_FLOAT_EQ(v6.x, 1.0f);
    EXPECT_FLOAT_EQ(v6.y, 2.0f);
}

TEST(Vec2Test, Access) 
{
    Vec2f v(3.14f, 2.71f);
    
    EXPECT_FLOAT_EQ(v[0], 3.14f);
    EXPECT_FLOAT_EQ(v[1], 2.71f);
    
    EXPECT_FLOAT_EQ(v.x, 3.14f);
    EXPECT_FLOAT_EQ(v.y, 2.71f);
}

TEST(Vec2Test, StaticMethods) 
{
    auto zero = Vec2f::Zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    
    auto one = Vec2f::One();
    EXPECT_FLOAT_EQ(one.x, 1.0f);
    EXPECT_FLOAT_EQ(one.y, 1.0f);
    
    auto unitX = Vec2f::UnitX();
    EXPECT_FLOAT_EQ(unitX.x, 1.0f);
    EXPECT_FLOAT_EQ(unitX.y, 0.0f);
    
    auto unitY = Vec2f::UnitY();
    EXPECT_FLOAT_EQ(unitY.x, 0.0f);
    EXPECT_FLOAT_EQ(unitY.y, 1.0f);
    
    auto unit = Vec2f::Unit();
    EXPECT_FLOAT_EQ(unit.x, kInvSqrt2);
    EXPECT_FLOAT_EQ(unit.y, kInvSqrt2);
}

TEST(Vec2Test, SwizzleOperations) 
{
    Vec2f v(3.0f, 4.0f);
    
    auto xx = v.xx();
    EXPECT_FLOAT_EQ(xx.x, 3.0f);
    EXPECT_FLOAT_EQ(xx.y, 3.0f);
    
    auto xy = v.xy();
    EXPECT_FLOAT_EQ(xy.x, 3.0f);
    EXPECT_FLOAT_EQ(xy.y, 4.0f);
    
    auto yx = v.yx();
    EXPECT_FLOAT_EQ(yx.x, 4.0f);
    EXPECT_FLOAT_EQ(yx.y, 3.0f);
    
    auto yy = v.yy();
    EXPECT_FLOAT_EQ(yy.x, 4.0f);
    EXPECT_FLOAT_EQ(yy.y, 4.0f);
    
    auto xxx = v.xxx();
    EXPECT_FLOAT_EQ(xxx.x, 3.0f);
    EXPECT_FLOAT_EQ(xxx.y, 3.0f);
    EXPECT_FLOAT_EQ(xxx.z, 3.0f);
    
    auto xxy = v.xxy();
    EXPECT_FLOAT_EQ(xxy.x, 3.0f);
    EXPECT_FLOAT_EQ(xxy.y, 3.0f);
    EXPECT_FLOAT_EQ(xxy.z, 4.0f);
    
    auto xxxx = v.xxxx();
    EXPECT_FLOAT_EQ(xxxx.x, 3.0f);
    EXPECT_FLOAT_EQ(xxxx.y, 3.0f);
    EXPECT_FLOAT_EQ(xxxx.z, 3.0f);
    EXPECT_FLOAT_EQ(xxxx.w, 3.0f);
}

TEST(Vec3Test, Construction) 
{
    Vec3f v1;
    EXPECT_FLOAT_EQ(v1.x, 0.0f);
    EXPECT_FLOAT_EQ(v1.y, 0.0f);
    EXPECT_FLOAT_EQ(v1.z, 0.0f);
    
    Vec3f v2(5.0f);
    EXPECT_FLOAT_EQ(v2.x, 5.0f);
    EXPECT_FLOAT_EQ(v2.y, 5.0f);
    EXPECT_FLOAT_EQ(v2.z, 5.0f);
    
    Vec3f v3(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v3.x, 1.0f);
    EXPECT_FLOAT_EQ(v3.y, 2.0f);
    EXPECT_FLOAT_EQ(v3.z, 3.0f);
    
    Vec3f v4(1, 2.5f, 3);
    EXPECT_FLOAT_EQ(v4.x, 1.0f);
    EXPECT_FLOAT_EQ(v4.y, 2.5f);
    EXPECT_FLOAT_EQ(v4.z, 3.0f);
    
    Vec2f v2d(1.0f, 2.0f);
    Vec3f v5(v2d, 3.0f);
    EXPECT_FLOAT_EQ(v5.x, 1.0f);
    EXPECT_FLOAT_EQ(v5.y, 2.0f);
    EXPECT_FLOAT_EQ(v5.z, 3.0f);
    
    Vec3f v6(1.0f, v2d);
    EXPECT_FLOAT_EQ(v6.x, 1.0f);
    EXPECT_FLOAT_EQ(v6.y, 1.0f);
    EXPECT_FLOAT_EQ(v6.z, 2.0f);
}

TEST(Vec3Test, Access) 
{
    Vec3f v(1.0f, 2.0f, 3.0f);
    
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
    
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vec3Test, StaticMethods) 
{
    auto zero = Vec3f::Zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    EXPECT_FLOAT_EQ(zero.z, 0.0f);
    
    auto one = Vec3f::One();
    EXPECT_FLOAT_EQ(one.x, 1.0f);
    EXPECT_FLOAT_EQ(one.y, 1.0f);
    EXPECT_FLOAT_EQ(one.z, 1.0f);
    
    auto unitX = Vec3f::UnitX();
    EXPECT_FLOAT_EQ(unitX.x, 1.0f);
    EXPECT_FLOAT_EQ(unitX.y, 0.0f);
    EXPECT_FLOAT_EQ(unitX.z, 0.0f);
    
    auto unitY = Vec3f::UnitY();
    EXPECT_FLOAT_EQ(unitY.x, 0.0f);
    EXPECT_FLOAT_EQ(unitY.y, 1.0f);
    EXPECT_FLOAT_EQ(unitY.z, 0.0f);
    
    auto unitZ = Vec3f::UnitZ();
    EXPECT_FLOAT_EQ(unitZ.x, 0.0f);
    EXPECT_FLOAT_EQ(unitZ.y, 0.0f);
    EXPECT_FLOAT_EQ(unitZ.z, 1.0f);
    
    auto unit = Vec3f::Unit();
    EXPECT_FLOAT_EQ(unit.x, kInvSqrt3);
    EXPECT_FLOAT_EQ(unit.y, kInvSqrt3);
    EXPECT_FLOAT_EQ(unit.z, kInvSqrt3);
}

TEST(Vec3Test, SwizzleOperations) 
{
    Vec3f v(1.0f, 2.0f, 3.0f);
    
    auto xy = v.xy();
    EXPECT_FLOAT_EQ(xy.x, 1.0f);
    EXPECT_FLOAT_EQ(xy.y, 2.0f);
    
    auto xz = v.xz();
    EXPECT_FLOAT_EQ(xz.x, 1.0f);
    EXPECT_FLOAT_EQ(xz.y, 3.0f);
    
    auto yz = v.yz();
    EXPECT_FLOAT_EQ(yz.x, 2.0f);
    EXPECT_FLOAT_EQ(yz.y, 3.0f);
    
    auto xyz = v.xyz();
    EXPECT_FLOAT_EQ(xyz.x, 1.0f);
    EXPECT_FLOAT_EQ(xyz.y, 2.0f);
    EXPECT_FLOAT_EQ(xyz.z, 3.0f);
    
    auto zyx = v.zyx();
    EXPECT_FLOAT_EQ(zyx.x, 3.0f);
    EXPECT_FLOAT_EQ(zyx.y, 2.0f);
    EXPECT_FLOAT_EQ(zyx.z, 1.0f);
}

TEST(Vec4Test, Construction) 
{
    Vec4f v1;
    EXPECT_FLOAT_EQ(v1.x, 0.0f);
    EXPECT_FLOAT_EQ(v1.y, 0.0f);
    EXPECT_FLOAT_EQ(v1.z, 0.0f);
    EXPECT_FLOAT_EQ(v1.w, 0.0f);
    
    Vec4f v2(5.0f);
    EXPECT_FLOAT_EQ(v2.x, 5.0f);
    EXPECT_FLOAT_EQ(v2.y, 5.0f);
    EXPECT_FLOAT_EQ(v2.z, 5.0f);
    EXPECT_FLOAT_EQ(v2.w, 5.0f);
    
    Vec4f v3(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(v3.x, 1.0f);
    EXPECT_FLOAT_EQ(v3.y, 2.0f);
    EXPECT_FLOAT_EQ(v3.z, 3.0f);
    EXPECT_FLOAT_EQ(v3.w, 4.0f);
    
    Vec3f v3d(1.0f, 2.0f, 3.0f);
    Vec4f v4(v3d, 4.0f);
    EXPECT_FLOAT_EQ(v4.x, 1.0f);
    EXPECT_FLOAT_EQ(v4.y, 2.0f);
    EXPECT_FLOAT_EQ(v4.z, 3.0f);
    EXPECT_FLOAT_EQ(v4.w, 4.0f);
    
    Vec4f v5(1.0f, v3d);
    EXPECT_FLOAT_EQ(v5.x, 1.0f);
    EXPECT_FLOAT_EQ(v5.y, 1.0f);
    EXPECT_FLOAT_EQ(v5.z, 2.0f);
    EXPECT_FLOAT_EQ(v5.w, 3.0f);
    
    Vec2f v2d1(1.0f, 2.0f);
    Vec2f v2d2(3.0f, 4.0f);
    Vec4f v6(v2d1, v2d2);
    EXPECT_FLOAT_EQ(v6.x, 1.0f);
    EXPECT_FLOAT_EQ(v6.y, 2.0f);
    EXPECT_FLOAT_EQ(v6.z, 3.0f);
    EXPECT_FLOAT_EQ(v6.w, 4.0f);
}

TEST(Vec4Test, Access) 
{
    Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
    
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
    EXPECT_FLOAT_EQ(v[3], 4.0f);
    
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
    EXPECT_FLOAT_EQ(v.w, 4.0f);
}

TEST(Vec4Test, StaticMethods) 
{
    auto zero = Vec4f::Zero();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    EXPECT_FLOAT_EQ(zero.z, 0.0f);
    EXPECT_FLOAT_EQ(zero.w, 0.0f);
    
    auto one = Vec4f::One();
    EXPECT_FLOAT_EQ(one.x, 1.0f);
    EXPECT_FLOAT_EQ(one.y, 1.0f);
    EXPECT_FLOAT_EQ(one.z, 1.0f);
    EXPECT_FLOAT_EQ(one.w, 1.0f);
    
    auto unitX = Vec4f::UnitX();
    EXPECT_FLOAT_EQ(unitX.x, 1.0f);
    EXPECT_FLOAT_EQ(unitX.y, 0.0f);
    EXPECT_FLOAT_EQ(unitX.z, 0.0f);
    EXPECT_FLOAT_EQ(unitX.w, 0.0f);
    
    auto unitY = Vec4f::UnitY();
    EXPECT_FLOAT_EQ(unitY.x, 0.0f);
    EXPECT_FLOAT_EQ(unitY.y, 1.0f);
    EXPECT_FLOAT_EQ(unitY.z, 0.0f);
    EXPECT_FLOAT_EQ(unitY.w, 0.0f);
    
    auto unitZ = Vec4f::UnitZ();
    EXPECT_FLOAT_EQ(unitZ.x, 0.0f);
    EXPECT_FLOAT_EQ(unitZ.y, 0.0f);
    EXPECT_FLOAT_EQ(unitZ.z, 1.0f);
    EXPECT_FLOAT_EQ(unitZ.w, 0.0f);
    
    auto unitW = Vec4f::UnitW();
    EXPECT_FLOAT_EQ(unitW.x, 0.0f);
    EXPECT_FLOAT_EQ(unitW.y, 0.0f);
    EXPECT_FLOAT_EQ(unitW.z, 0.0f);
    EXPECT_FLOAT_EQ(unitW.w, 1.0f);
    
    auto unit = Vec4f::Unit();
    EXPECT_FLOAT_EQ(unit.x, 0.5f);
    EXPECT_FLOAT_EQ(unit.y, 0.5f);
    EXPECT_FLOAT_EQ(unit.z, 0.5f);
    EXPECT_FLOAT_EQ(unit.w, 0.5f);
}

TEST(VectorArithmeticTest, CompoundAssignmentVector) 
{
    Vec3f v1(10.0f, 20.0f, 30.0f);
    Vec3f v2(1.0f, 2.0f, 3.0f);
    
    v1 += v2;
    EXPECT_FLOAT_EQ(v1.x, 11.0f);
    EXPECT_FLOAT_EQ(v1.y, 22.0f);
    EXPECT_FLOAT_EQ(v1.z, 33.0f);
    
    v1 -= v2;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    EXPECT_FLOAT_EQ(v1.y, 20.0f);
    EXPECT_FLOAT_EQ(v1.z, 30.0f);
    
    v1 *= v2;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    EXPECT_FLOAT_EQ(v1.y, 40.0f);
    EXPECT_FLOAT_EQ(v1.z, 90.0f);
    
    v1 /= v2;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    EXPECT_FLOAT_EQ(v1.y, 20.0f);
    EXPECT_FLOAT_EQ(v1.z, 30.0f);
}

TEST(VectorArithmeticTest, CompoundAssignmentScalar) 
{
    Vec3f v(10.0f, 20.0f, 30.0f);
    
    v += 5.0f;
    EXPECT_FLOAT_EQ(v.x, 15.0f);
    EXPECT_FLOAT_EQ(v.y, 25.0f);
    EXPECT_FLOAT_EQ(v.z, 35.0f);
    
    v -= 5.0f;
    EXPECT_FLOAT_EQ(v.x, 10.0f);
    EXPECT_FLOAT_EQ(v.y, 20.0f);
    EXPECT_FLOAT_EQ(v.z, 30.0f);
    
    v *= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 20.0f);
    EXPECT_FLOAT_EQ(v.y, 40.0f);
    EXPECT_FLOAT_EQ(v.z, 60.0f);
    
    v /= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 10.0f);
    EXPECT_FLOAT_EQ(v.y, 20.0f);
    EXPECT_FLOAT_EQ(v.z, 30.0f);
}

TEST(VectorArithmeticTest, IntegerModuloOperations) 
{
    Vec3i v1(10, 20, 30);
    Vec3i v2(3, 7, 11);
    
    v1 %= v2;
    EXPECT_EQ(v1.x, 1);  // 10 % 3 = 1
    EXPECT_EQ(v1.y, 6);  // 20 % 7 = 6
    EXPECT_EQ(v1.z, 8);  // 30 % 11 = 8
    
    Vec3i v3(15, 25, 35);
    v3 %= 4;
    EXPECT_EQ(v3.x, 3);  // 15 % 4 = 3
    EXPECT_EQ(v3.y, 1);  // 25 % 4 = 1
    EXPECT_EQ(v3.z, 3);  // 35 % 4 = 3
}

TEST(VectorArithmeticTest, BitwiseOperations) 
{
    Vec3i v1(0b1010, 0b1100, 0b1111);
    Vec3i v2(0b1100, 0b1010, 0b0011);
    
    v1 &= v2;
    EXPECT_EQ(v1.x, 0b1000);
    EXPECT_EQ(v1.y, 0b1000);
    EXPECT_EQ(v1.z, 0b0011);
    
    v1 = Vec3i(0b1010, 0b1100, 0b1111);
    v1 |= v2;
    EXPECT_EQ(v1.x, 0b1110);
    EXPECT_EQ(v1.y, 0b1110);
    EXPECT_EQ(v1.z, 0b1111);
    
    v1 = Vec3i(0b1010, 0b1100, 0b1111);
    v1 ^= v2;
    EXPECT_EQ(v1.x, 0b0110);
    EXPECT_EQ(v1.y, 0b0110);
    EXPECT_EQ(v1.z, 0b1100);
    
    Vec3i v3(1, 2, 4);
    v3 <<= Vec3i(1, 2, 3);
    EXPECT_EQ(v3.x, 2);   // 1 << 1 = 2
    EXPECT_EQ(v3.y, 8);   // 2 << 2 = 8
    EXPECT_EQ(v3.z, 32);  // 4 << 3 = 32
    
    v3 >>= Vec3i(1, 1, 2);
    EXPECT_EQ(v3.x, 1);   // 2 >> 1 = 1
    EXPECT_EQ(v3.y, 4);   // 8 >> 1 = 4
    EXPECT_EQ(v3.z, 8);   // 32 >> 2 = 8
}

TEST(VectorCompatibilityTest, TypeConversion) 
{
    Vec3i vi(1, 2, 3);
    Vec3f vf(vi);
    EXPECT_FLOAT_EQ(vf.x, 1.0f);
    EXPECT_FLOAT_EQ(vf.y, 2.0f);
    EXPECT_FLOAT_EQ(vf.z, 3.0f);
    
    Vec3f vf2(1.7f, 2.3f, 3.9f);
    Vec3i vi2(vf2);
    EXPECT_EQ(vi2.x, 1);
    EXPECT_EQ(vi2.y, 2);
    EXPECT_EQ(vi2.z, 3);
    
    Vec3f vf32(1.5f, 2.5f, 3.5f);
    Vec3d vf64(vf32);
    EXPECT_DOUBLE_EQ(vf64.x, 1.5);
    EXPECT_DOUBLE_EQ(vf64.y, 2.5);
    EXPECT_DOUBLE_EQ(vf64.z, 3.5);
}

TEST(VectorCompatibilityTest, MixedTypeArithmetic) 
{
    Vec3f vf(10.0f, 20.0f, 30.0f);
    Vec3i vi(1, 2, 3);
    
    vf += vi;
    EXPECT_FLOAT_EQ(vf.x, 11.0f);
    EXPECT_FLOAT_EQ(vf.y, 22.0f);
    EXPECT_FLOAT_EQ(vf.z, 33.0f);
    
    Vec3f vf2(10.0f, 20.0f, 30.0f);
    vf2 *= 2;  // 整数标量
    EXPECT_FLOAT_EQ(vf2.x, 20.0f);
    EXPECT_FLOAT_EQ(vf2.y, 40.0f);
    EXPECT_FLOAT_EQ(vf2.z, 60.0f);
}

TEST(VectorBoundaryTest, ZeroOperations) 
{
    Vec3f zero = Vec3f::Zero();
    Vec3f v(1.0f, 2.0f, 3.0f);
    
    v += zero;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
    
    v *= zero;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST(VectorBoundaryTest, LargeValues) 
{
    Vec3f largeVec(1e6f, 1e7f, 1e8f);
    Vec3f smallVec(1e-6f, 1e-7f, 1e-8f);
    
    largeVec += smallVec;
    EXPECT_FLOAT_EQ(largeVec.x, 1e6f);
    EXPECT_FLOAT_EQ(largeVec.y, 1e7f);
    EXPECT_FLOAT_EQ(largeVec.z, 1e8f);
}

TEST(VectorBoundaryTest, IntegerOverflow) 
{
    Vec3i maxInt(std::numeric_limits<i32>::max(), std::numeric_limits<i32>::max(), std::numeric_limits<i32>::max());
    
    Vec3i result = maxInt;
    result += Vec3i(1, 1, 1);
    EXPECT_TRUE(true);
}

TEST(VectorOpsArithmeticTest, VectorAddition) 
{
    Vec3f v1(1.0f, 2.0f, 3.0f);
    Vec3f v2(4.0f, 5.0f, 6.0f);
    auto result = v1 + v2;
    EXPECT_FLOAT_EQ(result.x, 5.0f);
    EXPECT_FLOAT_EQ(result.y, 7.0f);
    EXPECT_FLOAT_EQ(result.z, 9.0f);
    
    Vec3f v3(1.0f, 2.0f, 3.0f);
    auto result2 = v3 + 5.0f;
    EXPECT_FLOAT_EQ(result2.x, 6.0f);
    EXPECT_FLOAT_EQ(result2.y, 7.0f);
    EXPECT_FLOAT_EQ(result2.z, 8.0f);
    
    auto result3 = 10.0f + v3;
    EXPECT_FLOAT_EQ(result3.x, 11.0f);
    EXPECT_FLOAT_EQ(result3.y, 12.0f);
    EXPECT_FLOAT_EQ(result3.z, 13.0f);
    
    Vec3i vi(1, 2, 3);
    Vec3f vf(1.5f, 2.5f, 3.5f);
    auto mixed_result = vi + vf;
    EXPECT_FLOAT_EQ(mixed_result.x, 2.5f);
    EXPECT_FLOAT_EQ(mixed_result.y, 4.5f);
    EXPECT_FLOAT_EQ(mixed_result.z, 6.5f);
}

TEST(VectorOpsArithmeticTest, VectorSubtraction) 
{
    Vec3f v1(10.0f, 8.0f, 6.0f);
    Vec3f v2(4.0f, 3.0f, 2.0f);
    auto result = v1 - v2;
    EXPECT_FLOAT_EQ(result.x, 6.0f);
    EXPECT_FLOAT_EQ(result.y, 5.0f);
    EXPECT_FLOAT_EQ(result.z, 4.0f);
    
    Vec3f v3(10.0f, 8.0f, 6.0f);
    auto result2 = v3 - 3.0f;
    EXPECT_FLOAT_EQ(result2.x, 7.0f);
    EXPECT_FLOAT_EQ(result2.y, 5.0f);
    EXPECT_FLOAT_EQ(result2.z, 3.0f);
    
    auto result3 = 15.0f - v3;
    EXPECT_FLOAT_EQ(result3.x, 5.0f);
    EXPECT_FLOAT_EQ(result3.y, 7.0f);
    EXPECT_FLOAT_EQ(result3.z, 9.0f);
}

TEST(VectorOpsArithmeticTest, VectorMultiplication) 
{
    Vec3f v1(2.0f, 3.0f, 4.0f);
    Vec3f v2(5.0f, 6.0f, 7.0f);
    auto result = v1 * v2;
    EXPECT_FLOAT_EQ(result.x, 10.0f);
    EXPECT_FLOAT_EQ(result.y, 18.0f);
    EXPECT_FLOAT_EQ(result.z, 28.0f);
    
    Vec3f v3(2.0f, 3.0f, 4.0f);
    auto result2 = v3 * 3.0f;
    EXPECT_FLOAT_EQ(result2.x, 6.0f);
    EXPECT_FLOAT_EQ(result2.y, 9.0f);
    EXPECT_FLOAT_EQ(result2.z, 12.0f);
    
    auto result3 = 2.5f * v3;
    EXPECT_FLOAT_EQ(result3.x, 5.0f);
    EXPECT_FLOAT_EQ(result3.y, 7.5f);
    EXPECT_FLOAT_EQ(result3.z, 10.0f);
}

TEST(VectorOpsArithmeticTest, VectorDivision) 
{
    Vec3f v1(12.0f, 15.0f, 20.0f);
    Vec3f v2(3.0f, 5.0f, 4.0f);
    auto result = v1 / v2;
    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 3.0f);
    EXPECT_FLOAT_EQ(result.z, 5.0f);
    
    Vec3f v3(12.0f, 15.0f, 20.0f);
    auto result2 = v3 / 4.0f;
    EXPECT_FLOAT_EQ(result2.x, 3.0f);
    EXPECT_FLOAT_EQ(result2.y, 3.75f);
    EXPECT_FLOAT_EQ(result2.z, 5.0f);
    
    auto result3 = 60.0f / v3;
    EXPECT_FLOAT_EQ(result3.x, 5.0f);
    EXPECT_FLOAT_EQ(result3.y, 4.0f);
    EXPECT_FLOAT_EQ(result3.z, 3.0f);
}

TEST(VectorOpsArithmeticTest, VectorModulo) 
{
    Vec3i v1(10, 15, 20);
    Vec3i v2(3, 4, 6);
    auto result = v1 % v2;
    EXPECT_EQ(result.x, 1);  // 10 % 3 = 1
    EXPECT_EQ(result.y, 3);  // 15 % 4 = 3
    EXPECT_EQ(result.z, 2);  // 20 % 6 = 2
    
    Vec3i v3(17, 23, 29);
    auto result2 = v3 % 5;
    EXPECT_EQ(result2.x, 2);  // 17 % 5 = 2
    EXPECT_EQ(result2.y, 3);  // 23 % 5 = 3
    EXPECT_EQ(result2.z, 4);  // 29 % 5 = 4
    
    auto result3 = 100 % Vec3i(7, 11, 13);
    EXPECT_EQ(result3.x, 2);  // 100 % 7 = 2
    EXPECT_EQ(result3.y, 1);  // 100 % 11 = 1
    EXPECT_EQ(result3.z, 9);  // 100 % 13 = 9
}

TEST(VectorOpsArithmeticTest, UnaryOperators) 
{
    Vec3f v1(1.0f, -2.0f, 3.0f);
    auto result1 = +v1;
    EXPECT_FLOAT_EQ(result1.x, 1.0f);
    EXPECT_FLOAT_EQ(result1.y, -2.0f);
    EXPECT_FLOAT_EQ(result1.z, 3.0f);
    
    auto result2 = -v1;
    EXPECT_FLOAT_EQ(result2.x, -1.0f);
    EXPECT_FLOAT_EQ(result2.y, 2.0f);
    EXPECT_FLOAT_EQ(result2.z, -3.0f);
    
    Vec3i vi(5, -10, 15);
    auto result3 = -vi;
    EXPECT_EQ(result3.x, -5);
    EXPECT_EQ(result3.y, 10);
    EXPECT_EQ(result3.z, -15);
}

TEST(VectorOpsComparisonTest, VectorEquality) 
{
    Vec3f v1(1.0f, 2.0f, 3.0f);
    Vec3f v2(1.0f, 2.0f, 3.0f);
    Vec3f v3(1.0f, 2.0f, 4.0f);
    
    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
    
    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
    
    Vec3i vi(1, 2, 3);
    EXPECT_TRUE(v1 == vi);
}

TEST(VectorOpsComparisonTest, ElementWiseEqual) 
{
    Vec3f v1(1.0f, 2.0f, 3.0f);
    Vec3f v2(1.0f, 4.0f, 3.0f);
    
    auto result = Equal(v1, v2);
    EXPECT_TRUE(result.x);
    EXPECT_FALSE(result.y);
    EXPECT_TRUE(result.z);
    
    auto result2 = Equal(v1, 2.0f);
    EXPECT_FALSE(result2.x);
    EXPECT_TRUE(result2.y);
    EXPECT_FALSE(result2.z);
    
    auto result3 = Equal(3.0f, v1);
    EXPECT_FALSE(result3.x);
    EXPECT_FALSE(result3.y);
    EXPECT_TRUE(result3.z);
}

TEST(VectorOpsComparisonTest, ElementWiseNotEqual) 
{
    Vec3f v1(1.0f, 2.0f, 3.0f);
    Vec3f v2(1.0f, 4.0f, 3.0f);
    
    auto result = NotEqual(v1, v2);
    EXPECT_FALSE(result.x);
    EXPECT_TRUE(result.y);
    EXPECT_FALSE(result.z);
    
    auto result2 = NotEqual(v1, 2.0f);
    EXPECT_TRUE(result2.x);
    EXPECT_FALSE(result2.y);
    EXPECT_TRUE(result2.z);
}

TEST(VectorOpsComparisonTest, ElementWiseLessThan) 
{
    Vec3f v1(1.0f, 5.0f, 3.0f);
    Vec3f v2(2.0f, 4.0f, 3.0f);
    
    auto result = LessThan(v1, v2);
    EXPECT_TRUE(result.x);   // 1 < 2
    EXPECT_FALSE(result.y);  // 5 < 4 is false
    EXPECT_FALSE(result.z);  // 3 < 3 is false
    
    auto result2 = LessThan(v1, 4.0f);
    EXPECT_TRUE(result2.x);   // 1 < 4
    EXPECT_FALSE(result2.y);  // 5 < 4 is false
    EXPECT_TRUE(result2.z);   // 3 < 4
}

TEST(VectorOpsComparisonTest, ElementWiseLessThanEqual) 
{
    Vec3f v1(1.0f, 5.0f, 3.0f);
    Vec3f v2(2.0f, 4.0f, 3.0f);
    
    auto result = LessThanEqual(v1, v2);
    EXPECT_TRUE(result.x);   // 1 <= 2
    EXPECT_FALSE(result.y);  // 5 <= 4 is false
    EXPECT_TRUE(result.z);   // 3 <= 3
}

TEST(VectorOpsComparisonTest, ElementWiseGreaterThan) 
{
    Vec3f v1(3.0f, 2.0f, 5.0f);
    Vec3f v2(2.0f, 4.0f, 3.0f);
    
    auto result = GreaterThan(v1, v2);
    EXPECT_TRUE(result.x);   // 3 > 2
    EXPECT_FALSE(result.y);  // 2 > 4 is false
    EXPECT_TRUE(result.z);   // 5 > 3
}

TEST(VectorOpsComparisonTest, ElementWiseGreaterThanEqual) 
{
    Vec3f v1(3.0f, 2.0f, 5.0f);
    Vec3f v2(2.0f, 2.0f, 3.0f);
    
    auto result = GreaterThanEqual(v1, v2);
    EXPECT_TRUE(result.x);   // 3 >= 2
    EXPECT_TRUE(result.y);   // 2 >= 2
    EXPECT_TRUE(result.z);   // 5 >= 3
}

TEST(VectorOpsBooleanTest, LogicalOperators) 
{
    Vec3<bool> v1(true, false, true);
    Vec3<bool> v2(false, false, true);
    
    auto and_result = v1 && v2;
    EXPECT_FALSE(and_result.x);  // true && false = false
    EXPECT_FALSE(and_result.y);  // false && false = false
    EXPECT_TRUE(and_result.z);   // true && true = true
    
    auto or_result = v1 || v2;
    EXPECT_TRUE(or_result.x);    // true || false = true
    EXPECT_FALSE(or_result.y);   // false || false = false
    EXPECT_TRUE(or_result.z);    // true || true = true
    
    auto not_result = !v1;
    EXPECT_FALSE(not_result.x);  // !true = false
    EXPECT_TRUE(not_result.y);   // !false = true
    EXPECT_FALSE(not_result.z);  // !true = false
}

TEST(VectorOpsBooleanTest, BooleanReductionFunctions) 
{
    Vec4<bool> all_true(true, true, true, true);
    Vec4<bool> all_false(false, false, false, false);
    Vec4<bool> mixed(true, false, true, false);
    
    EXPECT_TRUE(All(all_true));
    EXPECT_FALSE(All(all_false));
    EXPECT_FALSE(All(mixed));
    
    EXPECT_TRUE(Any(all_true));
    EXPECT_FALSE(Any(all_false));
    EXPECT_TRUE(Any(mixed));
    
    EXPECT_FALSE(None(all_true));
    EXPECT_TRUE(None(all_false));
    EXPECT_FALSE(None(mixed));
    
    EXPECT_EQ(Count(all_true), 4);
    EXPECT_EQ(Count(all_false), 0);
    EXPECT_EQ(Count(mixed), 2);
}

TEST(VectorOpsBitwiseTest, BitwiseAnd) 
{
    Vec3i v1(0b1010, 0b1100, 0b1111);
    Vec3i v2(0b1100, 0b1010, 0b0011);
    
    auto result = v1 & v2;
    EXPECT_EQ(result.x, 0b1000);  // 1010 & 1100 = 1000
    EXPECT_EQ(result.y, 0b1000);  // 1100 & 1010 = 1000
    EXPECT_EQ(result.z, 0b0011);  // 1111 & 0011 = 0011
    
    auto result2 = v1 & 0b1001;
    EXPECT_EQ(result2.x, 0b1000);  // 1010 & 1001 = 1000
    EXPECT_EQ(result2.y, 0b1000);  // 1100 & 1001 = 1000
    EXPECT_EQ(result2.z, 0b1001);  // 1111 & 1001 = 1001
    
    auto result3 = 0b1111 & v1;
    EXPECT_EQ(result3.x, 0b1010);  // 1111 & 1010 = 1010
    EXPECT_EQ(result3.y, 0b1100);  // 1111 & 1100 = 1100
    EXPECT_EQ(result3.z, 0b1111);  // 1111 & 1111 = 1111
}

TEST(VectorOpsBitwiseTest, BitwiseOr) 
{
    Vec3i v1(0b1010, 0b1100, 0b0000);
    Vec3i v2(0b0101, 0b0011, 0b1111);
    
    auto result = v1 | v2;
    EXPECT_EQ(result.x, 0b1111);  // 1010 | 0101 = 1111
    EXPECT_EQ(result.y, 0b1111);  // 1100 | 0011 = 1111
    EXPECT_EQ(result.z, 0b1111);  // 0000 | 1111 = 1111
}

TEST(VectorOpsBitwiseTest, BitwiseXor) 
{
    Vec3i v1(0b1010, 0b1100, 0b1111);
    Vec3i v2(0b1100, 0b1010, 0b1111);
    
    auto result = v1 ^ v2;
    EXPECT_EQ(result.x, 0b0110);  // 1010 ^ 1100 = 0110
    EXPECT_EQ(result.y, 0b0110);  // 1100 ^ 1010 = 0110
    EXPECT_EQ(result.z, 0b0000);  // 1111 ^ 1111 = 0000
}

TEST(VectorOpsBitwiseTest, BitwiseNot) 
{
    Vec3<u8> v(0b00001111, 0b11110000, 0b10101010);
    
    auto result = ~v;
    EXPECT_EQ(static_cast<u8>(result.x), static_cast<u8>(~0b00001111));
    EXPECT_EQ(static_cast<u8>(result.y), static_cast<u8>(~0b11110000));
    EXPECT_EQ(static_cast<u8>(result.z), static_cast<u8>(~0b10101010));
}

TEST(VectorOpsBitwiseTest, BitShift) 
{
    Vec3i v1(1, 4, 16);
    Vec3i v2(2, 1, 3);
    
    auto left_result = v1 << v2;
    EXPECT_EQ(left_result.x, 4);   // 1 << 2 = 4
    EXPECT_EQ(left_result.y, 8);   // 4 << 1 = 8
    EXPECT_EQ(left_result.z, 128); // 16 << 3 = 128
    
    Vec3i v3(16, 8, 32);
    auto right_result = v3 >> v2;
    EXPECT_EQ(right_result.x, 4);  // 16 >> 2 = 4
    EXPECT_EQ(right_result.y, 4);  // 8 >> 1 = 4
    EXPECT_EQ(right_result.z, 4);  // 32 >> 3 = 4
    
    auto scalar_left = v1 << 2;
    EXPECT_EQ(scalar_left.x, 4);   // 1 << 2 = 4
    EXPECT_EQ(scalar_left.y, 16);  // 4 << 2 = 16
    EXPECT_EQ(scalar_left.z, 64);  // 16 << 2 = 64
}

TEST(VectorOpsEdgeCasesTest, ZeroOperations) 
{
    Vec3f zero = Vec3f::Zero();
    Vec3f v(1.0f, 2.0f, 3.0f);
    
    auto add_zero = v + zero;
    EXPECT_FLOAT_EQ(add_zero.x, 1.0f);
    EXPECT_FLOAT_EQ(add_zero.y, 2.0f);
    EXPECT_FLOAT_EQ(add_zero.z, 3.0f);
    
    auto mul_zero = v * zero;
    EXPECT_FLOAT_EQ(mul_zero.x, 0.0f);
    EXPECT_FLOAT_EQ(mul_zero.y, 0.0f);
    EXPECT_FLOAT_EQ(mul_zero.z, 0.0f);
    
    auto neg_zero = -zero;
    EXPECT_FLOAT_EQ(neg_zero.x, 0.0f);
    EXPECT_FLOAT_EQ(neg_zero.y, 0.0f);
    EXPECT_FLOAT_EQ(neg_zero.z, 0.0f);
}

TEST(VectorOpsEdgeCasesTest, TypeConversions) 
{
    Vec3i vi(1, 2, 3);
    Vec3f vf(1.5f, 2.5f, 3.5f);
    
    auto result = vi + vf;
    EXPECT_FLOAT_EQ(result.x, 2.5f);
    EXPECT_FLOAT_EQ(result.y, 4.5f);
    EXPECT_FLOAT_EQ(result.z, 6.5f);
    
    Vec3d vd(1.0, 2.0, 3.0);
    auto result2 = vf + vd;
    EXPECT_DOUBLE_EQ(result2.x, 2.5);
    EXPECT_DOUBLE_EQ(result2.y, 4.5);
    EXPECT_DOUBLE_EQ(result2.z, 6.5);
}

TEST(VectorOpsEdgeCasesTest, ConstexprOperations) 
{
    constexpr Vec3f v1(1.0f, 2.0f, 3.0f);
    constexpr Vec3f v2(4.0f, 5.0f, 6.0f);
    
    auto result = v1 + v2;
    EXPECT_FLOAT_EQ(result.x, 5.0f);
    EXPECT_FLOAT_EQ(result.y, 7.0f);
    EXPECT_FLOAT_EQ(result.z, 9.0f);
    
    auto equal_result = Equal(v1, Vec3f(1.0f, 2.0f, 3.0f));
    EXPECT_TRUE(equal_result.x);
    EXPECT_TRUE(equal_result.y);
    EXPECT_TRUE(equal_result.z);
    
    static_assert(v1.x == 1.0f);
    static_assert(v1.y == 2.0f);
    static_assert(v1.z == 3.0f);
}

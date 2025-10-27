/**
 * @File SharkTest.cpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/27
 * @Brief This file is part of Shark.
 */

#include <gtest/gtest.h>

#include <Shark/Shark.hpp>

using namespace SKM;

TEST(SharkTest, TestFunc)
{
    EXPECT_EQ(TestFunc(), 42);
}

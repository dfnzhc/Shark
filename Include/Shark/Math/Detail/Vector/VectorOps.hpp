/**
 * @File VectorOps.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/3
 * @Brief This file is part of Shark.
 */

#pragma once

#include "VectorType.hpp"

namespace SKT
{

    // ==================== 算术操作 ====================
    namespace Detail
    {
        struct AddOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a + b)
            {
                return a + b;
            }
        };

        struct SubOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a - b)
            {
                return a - b;
            }
        };

        struct MulOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a * b)
            {
                return a * b;
            }
        };

        struct DivOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a / b)
            {
                return a / b;
            }
        };

        struct ModOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a % b)
            {
                return a % b;
            }
        };

        struct UnaryPlusOp
        {
            template <typename T>
            static constexpr auto Apply(T a) -> decltype(+a)
            {
                return +a;
            }
        };

        struct UnaryMinusOp
        {
            template <typename T>
            static constexpr auto Apply(T a) -> decltype(-a)
            {
                return -a;
            }
        };
    } // namespace Detail

    // 向量 + 向量
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, std::common_type_t<T, U>> operator+(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ElementWiseOp<Detail::AddOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 + 标量
    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<T, S>> operator+(Vec<N, T> lhs, S rhs)
    {
        return Detail::ElementWiseOp<Detail::AddOp>::ApplyBinary(lhs, rhs);
    }

    // 标量 + 向量
    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<S, T>> operator+(S lhs, Vec<N, T> rhs)
    {
        return Detail::ElementWiseOp<Detail::AddOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 - 向量
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, std::common_type_t<T, U>> operator-(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ElementWiseOp<Detail::SubOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 - 标量
    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<T, S>> operator-(Vec<N, T> lhs, S rhs)
    {
        return Detail::ElementWiseOp<Detail::SubOp>::ApplyBinary(lhs, rhs);
    }

    // 标量 - 向量
    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<S, T>> operator-(S lhs, Vec<N, T> rhs)
    {
        return Detail::ElementWiseOp<Detail::SubOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 * 向量
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, std::common_type_t<T, U>> operator*(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ElementWiseOp<Detail::MulOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 * 标量
    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<T, S>> operator*(Vec<N, T> lhs, S rhs)
    {
        return Detail::ElementWiseOp<Detail::MulOp>::ApplyBinary(lhs, rhs);
    }

    // 标量 * 向量
    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<S, T>> operator*(S lhs, Vec<N, T> rhs)
    {
        return Detail::ElementWiseOp<Detail::MulOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 / 向量
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, std::common_type_t<T, U>> operator/(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ElementWiseOp<Detail::DivOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 / 标量
    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<T, S>> operator/(Vec<N, T> lhs, S rhs)
    {
        return Detail::ElementWiseOp<Detail::DivOp>::ApplyBinary(lhs, rhs);
    }

    // 标量 / 向量
    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, std::common_type_t<S, T>> operator/(S lhs, Vec<N, T> rhs)
    {
        return Detail::ElementWiseOp<Detail::DivOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 % 向量
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>> && Detail::IntVectorType<Vec<N, T>> && Detail::IntVectorType
        <Vec<N, U>>
    constexpr Vec<N, std::common_type_t<T, U>> operator%(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ElementWiseOp<Detail::ModOp>::ApplyBinary(lhs, rhs);
    }

    // 向量 % 标量
    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S> && Detail::IntVectorType<Vec<N, T>> && IntegralType<S>
    constexpr Vec<N, std::common_type_t<T, S>> operator%(Vec<N, T> lhs, S rhs)
    {
        return Detail::ElementWiseOp<Detail::ModOp>::ApplyBinary(lhs, rhs);
    }

    // 标量 % 向量
    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S> && IntegralType<S> && Detail::IntVectorType<Vec<N, T>>
    constexpr Vec<N, std::common_type_t<S, T>> operator%(S lhs, Vec<N, T> rhs)
    {
        return Detail::ElementWiseOp<Detail::ModOp>::ApplyBinary(lhs, rhs);
    }

    // 一元正号操作符
    template <int N, typename T>
        requires Detail::VectorType<Vec<N, T>>
    constexpr Vec<N, T> operator+(Vec<N, T> v)
    {
        return Detail::ElementWiseOp<Detail::UnaryPlusOp>::ApplyUnary(v);
    }

    // 一元负号操作符
    template <int N, typename T>
        requires Detail::VectorType<Vec<N, T>>
    constexpr Vec<N, T> operator-(Vec<N, T> v)
    {
        return Detail::ElementWiseOp<Detail::UnaryMinusOp>::ApplyUnary(v);
    }

    // ==================== 比较操作 ====================
    namespace Detail
    {
        struct EqualOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a == b;
            }
        };

        struct NotEqualOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a != b;
            }
        };

        struct LessOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a < b;
            }
        };

        struct LessEqualOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a <= b;
            }
        };

        struct GreaterOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a > b;
            }
        };

        struct GreaterEqualOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a >= b;
            }
        };

        struct LogicalAndOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a && b;
            }
        };

        struct LogicalOrOp
        {
            template <typename T, typename U>
            static constexpr bool Apply(T a, U b)
            {
                return a || b;
            }
        };

        struct LogicalNotOp
        {
            template <typename T>
            static constexpr bool Apply(T a)
            {
                return !a;
            }
        };

        template <typename Op>
        struct ComparisonOp
        {
            template <int N, typename T, typename U>
            static constexpr Vec<N, bool> ApplyElementWise(Vec<N, T> lhs, Vec<N, U> rhs)
            {
                Vec<N, bool> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs[i], rhs[i]);
                }
                return result;
            }

            template <int N, typename T, typename S>
            static constexpr Vec<N, bool> ApplyElementWise(Vec<N, T> lhs, S rhs)
            {
                Vec<N, bool> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs[i], rhs);
                }
                return result;
            }

            template <int N, typename S, typename T>
            static constexpr Vec<N, bool> ApplyElementWise(S lhs, Vec<N, T> rhs)
            {
                Vec<N, bool> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs, rhs[i]);
                }
                return result;
            }

            template <int N, typename T, typename U>
            static constexpr bool ApplyAll(Vec<N, T> lhs, Vec<N, U> rhs)
            {
                for (int i = 0; i < N; ++i)
                {
                    if (!Op::Apply(lhs[i], rhs[i]))
                        return false;
                }
                return true;
            }

            template <int N, typename T, typename U>
            static constexpr bool ApplyAny(Vec<N, T> lhs, Vec<N, U> rhs)
            {
                for (int i = 0; i < N; ++i)
                {
                    if (Op::Apply(lhs[i], rhs[i]))
                        return true;
                }
                return false;
            }
        };
    } // namespace Detail

    // 全元素相等
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr bool operator==(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::EqualOp>::ApplyAll(lhs, rhs);
    }

    // 至少一个元素不等
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr bool operator!=(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::NotEqualOp>::ApplyAny(lhs, rhs);
    }

    // 逐元素相等
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> Equal(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::EqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> Equal(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::EqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> Equal(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::EqualOp>::ApplyElementWise(lhs, rhs);
    }

    // 逐元素不等
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> NotEqual(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::NotEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> NotEqual(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::NotEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> NotEqual(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::NotEqualOp>::ApplyElementWise(lhs, rhs);
    }

    // 逐元素小于
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> LessThan(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::LessOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> LessThan(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::LessOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> LessThan(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::LessOp>::ApplyElementWise(lhs, rhs);
    }

    // 逐元素小于等于
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> LessThanEqual(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::LessEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> LessThanEqual(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::LessEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> LessThanEqual(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::LessEqualOp>::ApplyElementWise(lhs, rhs);
    }

    // 逐元素大于
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> GreaterThan(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> GreaterThan(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> GreaterThan(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterOp>::ApplyElementWise(lhs, rhs);
    }

    // 逐元素大于等于
    template <int N, typename T, typename U>
        requires Detail::CompatibleVectors<Vec<N, T>, Vec<N, U>>
    constexpr Vec<N, bool> GreaterThanEqual(Vec<N, T> lhs, Vec<N, U> rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> GreaterThanEqual(Vec<N, T> lhs, S rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterEqualOp>::ApplyElementWise(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::VectorScalarCompatible<Vec<N, T>, S>
    constexpr Vec<N, bool> GreaterThanEqual(S lhs, Vec<N, T> rhs)
    {
        return Detail::ComparisonOp<Detail::GreaterEqualOp>::ApplyElementWise(lhs, rhs);
    }

    // 布尔操作
    template <int N>
    constexpr Vec<N, bool> operator&&(Vec<N, bool> lhs, Vec<N, bool> rhs)
    {
        Vec<N, bool> result;
        for (int i = 0; i < N; ++i)
        {
            result[i] = lhs[i] && rhs[i];
        }
        return result;
    }

    template <int N>
    constexpr Vec<N, bool> operator||(Vec<N, bool> lhs, Vec<N, bool> rhs)
    {
        Vec<N, bool> result;
        for (int i = 0; i < N; ++i)
        {
            result[i] = lhs[i] || rhs[i];
        }
        return result;
    }

    template <int N>
    constexpr Vec<N, bool> operator!(Vec<N, bool> v)
    {
        Vec<N, bool> result;
        for (int i = 0; i < N; ++i)
        {
            result[i] = !v[i];
        }
        return result;
    }

    // 所有元素都为真
    template <int N>
    constexpr bool All(Vec<N, bool> v)
    {
        for (int i = 0; i < N; ++i)
        {
            if (!v[i])
                return false;
        }
        return true;
    }

    // 至少一个元素为真
    template <int N>
    constexpr bool Any(Vec<N, bool> v)
    {
        for (int i = 0; i < N; ++i)
        {
            if (v[i])
                return true;
        }
        return false;
    }

    // 没有元素为真
    template <int N>
    constexpr bool None(Vec<N, bool> v)
    {
        return !Any(v);
    }

    // 计算为真的元素数量
    template <int N>
    constexpr int Count(Vec<N, bool> v)
    {
        int count = 0;
        for (int i = 0; i < N; ++i)
        {
            if (v[i])
                ++count;
        }
        return count;
    }

    // ==================== 位运算操作符标签 ====================
    namespace Detail
    {
        struct BitwiseAndOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a & b)
            {
                return a & b;
            }
        };

        struct BitwiseOrOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a | b)
            {
                return a | b;
            }
        };

        struct BitwiseXorOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a ^ b)
            {
                return a ^ b;
            }
        };

        struct BitwiseNotOp
        {
            template <typename T>
            static constexpr auto Apply(T a) -> decltype(~a)
            {
                return ~a;
            }
        };

        struct LeftShiftOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a << b)
            {
                return a << b;
            }
        };

        struct RightShiftOp
        {
            template <typename T, typename U>
            static constexpr auto Apply(T a, U b) -> decltype(a >> b)
            {
                return a >> b;
            }
        };
    } // namespace Detail

    template <int N, typename T, typename U>
        requires Detail::BitwiseVectorCompatible<N, T, U>
    constexpr auto operator&(Vec<N, T> lhs, Vec<N, U> rhs)
        -> Vec<N, decltype(std::declval<T>() & std::declval<U>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseAndOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator&(Vec<N, T> lhs, S rhs)
        -> Vec<N, decltype(std::declval<T>() & std::declval<S>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseAndOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator&(S lhs, Vec<N, T> rhs)
        -> Vec<N, decltype(std::declval<S>() & std::declval<T>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseAndOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename U>
        requires Detail::BitwiseVectorCompatible<N, T, U>
    constexpr auto operator|(Vec<N, T> lhs, Vec<N, U> rhs)
        -> Vec<N, decltype(std::declval<T>() | std::declval<U>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseOrOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator|(Vec<N, T> lhs, S rhs)
        -> Vec<N, decltype(std::declval<T>() | std::declval<S>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseOrOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator|(S lhs, Vec<N, T> rhs)
        -> Vec<N, decltype(std::declval<S>() | std::declval<T>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseOrOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename U>
        requires Detail::BitwiseVectorCompatible<N, T, U>
    constexpr auto operator^(Vec<N, T> lhs, Vec<N, U> rhs)
        -> Vec<N, decltype(std::declval<T>() ^ std::declval<U>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseXorOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator^(Vec<N, T> lhs, S rhs)
        -> Vec<N, decltype(std::declval<T>() ^ std::declval<S>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseXorOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename S, typename T>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator^(S lhs, Vec<N, T> rhs)
        -> Vec<N, decltype(std::declval<S>() ^ std::declval<T>())>
    {
        return Detail::ElementWiseOp<Detail::BitwiseXorOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T>
        requires Detail::VectorType<Vec<N, T>> && IntegralType<T>
    constexpr auto operator~(Vec<N, T> v)
        -> Vec<N, decltype(~std::declval<T>())>
    {
        Vec<N, decltype(~std::declval<T>())> result;
        for (int i = 0; i < N; ++i)
        {
            result[i] = ~v[i];
        }
        return result;
    }

    template <int N, typename T, typename U>
        requires Detail::BitwiseVectorCompatible<N, T, U>
    constexpr auto operator<<(Vec<N, T> lhs, Vec<N, U> rhs)
        -> Vec<N, decltype(std::declval<T>() << std::declval<U>())>
    {
        return Detail::ElementWiseOp<Detail::LeftShiftOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator<<(Vec<N, T> lhs, S rhs)
        -> Vec<N, decltype(std::declval<T>() << std::declval<S>())>
    {
        return Detail::ElementWiseOp<Detail::LeftShiftOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename U>
        requires Detail::BitwiseVectorCompatible<N, T, U>
    constexpr auto operator>>(Vec<N, T> lhs, Vec<N, U> rhs)
        -> Vec<N, decltype(std::declval<T>() >> std::declval<U>())>
    {
        return Detail::ElementWiseOp<Detail::RightShiftOp>::ApplyBinary(lhs, rhs);
    }

    template <int N, typename T, typename S>
        requires Detail::BitwiseVectorScalarCompatible<N, T, S>
    constexpr auto operator>>(Vec<N, T> lhs, S rhs)
        -> Vec<N, decltype(std::declval<T>() >> std::declval<S>())>
    {
        return Detail::ElementWiseOp<Detail::RightShiftOp>::ApplyBinary(lhs, rhs);
    }


} // namespace SKT

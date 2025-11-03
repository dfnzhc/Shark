/**
 * @File VectorCore.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/3
 * @Brief This file is part of Shark.
 */

#pragma once

#include "Shark/Core/Concepts.hpp"

namespace SKT
{
    // ==================== 前向声明 ====================

    template <int N, typename T>
    struct Vec;

    template <typename T>
    using Vec1 = Vec<1, T>;
    template <typename T>
    using Vec2 = Vec<2, T>;
    template <typename T>
    using Vec3 = Vec<3, T>;
    template <typename T>
    using Vec4 = Vec<4, T>;

    #define SKT_DEFINE_VECTOR_TYPE(type, suffix)    \
    using Vec1##suffix = Vec1<type>;                \
    using Vec2##suffix = Vec2<type>;                \
    using Vec3##suffix = Vec3<type>;                \
    using Vec4##suffix = Vec4<type>

    SKT_DEFINE_VECTOR_TYPE(f32, f);
    SKT_DEFINE_VECTOR_TYPE(f64, d);
    SKT_DEFINE_VECTOR_TYPE(i32, i);
    SKT_DEFINE_VECTOR_TYPE(u32, u);

    SKT_DEFINE_VECTOR_TYPE(i8, i8);
    SKT_DEFINE_VECTOR_TYPE(u8, u8);
    SKT_DEFINE_VECTOR_TYPE(i16, i16);
    SKT_DEFINE_VECTOR_TYPE(u16, u16);
    SKT_DEFINE_VECTOR_TYPE(i64, i64);
    SKT_DEFINE_VECTOR_TYPE(u64, u64);

    #undef SKT_DEFINE_VECTOR_TYPE

    // 避免与 CUDA 内置类型冲突
    #if defined(SKT_CPU_CODE) && !defined(__CUDACC__)
    #define SKT_DEFINE_VECTOR_TYPE(name, type)  \
    using name##1 = Vec1<type>;                 \
    using name##2 = Vec2<type>;                 \
    using name##3 = Vec3<type>;                 \
    using name##4 = Vec4<type>

    SKT_DEFINE_VECTOR_TYPE(float, f32);
    SKT_DEFINE_VECTOR_TYPE(double, f64);
    SKT_DEFINE_VECTOR_TYPE(int, i32);
    SKT_DEFINE_VECTOR_TYPE(uint, u32);
    SKT_DEFINE_VECTOR_TYPE(bool, bool);

    #undef SKT_DEFINE_VECTOR_TYPE
    #endif

    namespace Detail
    {
        // ==================== 概念约束 ====================

        template <int N>
        concept ValidVectorDimension = (N >= 1 && N <= 4);

        template <typename T>
        concept VectorType = requires
        {
            typename T::ValueType;
            typename T::SelfType;
            typename T::BooleanType;
            { T::Dimension } -> std::convertible_to<int>;
            requires ValidVectorDimension<T::Dimension>;
            requires ArithType<typename T::ValueType>;
        };

        template <typename T>
        concept FloatVectorType = VectorType<T> && FloatType<typename T::ValueType>;

        template <typename T>
        concept IntVectorType = VectorType<T> && IntegralType<typename T::ValueType>;

        template <typename T>
        concept BoolVectorType = VectorType<T> && BoolType<typename T::ValueType>;

        template <typename T, typename U>
        concept VectorSameDimension = VectorType<T> && VectorType<U> && (T::Dimension == U::Dimension);

        template <typename T, typename U>
        concept CompatibleVectors = VectorSameDimension<T, U> &&
                ArithType<typename T::ValueType> && ArithType<typename U::ValueType>;

        template <typename T, typename S>
        concept VectorScalarCompatible = VectorType<T> && ArithType<S>;

        template <int N, typename T, typename U>
        concept BitwiseVectorCompatible = VectorType<Vec<N, T>> && VectorType<Vec<N, U>> &&
                BitwiseCompatible<T, U> && VectorSameDimension<Vec<N, T>, Vec<N, U>>;

        template <int N, typename T, typename S>
        concept BitwiseVectorScalarCompatible = VectorType<Vec<N, T>> && BitwiseCompatible<T, S>;


        // ==================== 操作策略 ====================

        template <typename Op>
        struct VectorOp
        {
            template <typename Derived, typename Rhs>
            SKT_FUNC static Derived& Apply(Derived& lhs, Rhs rhs)
            {
                if constexpr (VectorType<Rhs>)
                {
                    for (int i = 0; i < Min(Derived::Dimension, Rhs::Dimension); ++i)
                    {
                        lhs[i] = Op::Apply(lhs[i], rhs[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < Derived::Dimension; ++i)
                    {
                        lhs[i] = Op::Apply(lhs[i], rhs);
                    }
                }
                return lhs;
            }
        };

        // ==================== 操作实现 ====================

        struct AddAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a + b;
            }
        };

        struct SubAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a - b;
            }
        };

        struct MulAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a * b;
            }
        };

        struct DivAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a / b;
            }
        };

        struct ModAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a % b;
            }
        };

        struct AndAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a & b;
            }
        };

        struct OrAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a | b;
            }
        };

        struct XorAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a ^ b;
            }
        };

        struct LShiftAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a << b;
            }
        };

        struct RShiftAssignOp
        {
            template <typename T, typename U>
            SKT_FUNC static T Apply(T a, U b)
            {
                return a >> b;
            }
        };

        // ==================== 逐元素操作 ====================

        template <typename Op>
        struct ElementWiseOp
        {
            template <int N, typename T>
            SKT_FUNC static Vec<N, T> ApplyUnary(Vec<N, T> v)
            {
                Vec<N, T> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(v[i]);
                }
                return result;
            }

            template <int N, typename T, typename U>
            SKT_FUNC static Vec<N, std::common_type_t<T, U>> ApplyBinary(Vec<N, T> lhs, Vec<N, U> rhs)
            {
                using ResultType = std::common_type_t<T, U>;
                Vec<N, ResultType> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs[i], rhs[i]);
                }
                return result;
            }

            template <int N, typename T, typename S>
            SKT_FUNC static Vec<N, std::common_type_t<T, S>> ApplyBinary(Vec<N, T> lhs, S rhs)
            {
                using ResultType = std::common_type_t<T, S>;
                Vec<N, ResultType> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs[i], rhs);
                }
                return result;
            }

            template <int N, typename S, typename T>
            SKT_FUNC static Vec<N, std::common_type_t<S, T>> ApplyBinary(S lhs, Vec<N, T> rhs)
            {
                using ResultType = std::common_type_t<S, T>;
                Vec<N, ResultType> result;
                for (int i = 0; i < N; ++i)
                {
                    result[i] = Op::Apply(lhs, rhs[i]);
                }
                return result;
            }
        };

        // ==================== 向量基础类型 ====================

        template <int N, typename T, typename Derived>
        class VectorBase
        {
        public:
            using ValueType                = T;
            using SelfType                 = Derived;
            using BooleanType              = Vec<N, bool>;
            static constexpr int Dimension = N;

            // 获取派生类引用
            SKT_FUNC SelfType& derived() noexcept
            {
                return static_cast<SelfType&>(*this);
            }

            SKT_FUNC const SelfType& derived() const noexcept
            {
                return static_cast<const SelfType&>(*this);
            }

            // 算术赋值 - 向量
            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>>
            SKT_FUNC SelfType& operator+=(Vec<N, U> rhs)
            {
                return VectorOp<AddAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>>
            SKT_FUNC SelfType& operator-=(Vec<N, U> rhs)
            {
                return VectorOp<SubAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>>
            SKT_FUNC SelfType& operator*=(Vec<N, U> rhs)
            {
                return VectorOp<MulAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>>
            SKT_FUNC SelfType& operator/=(Vec<N, U> rhs)
            {
                return VectorOp<DivAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator%=(Vec<N, U> rhs)
            {
                return VectorOp<ModAssignOp>::Apply(derived(), rhs);
            }

            // 算术赋值 - 标量
            template <typename S> requires VectorScalarCompatible<SelfType, S>
            SKT_FUNC SelfType& operator+=(S scalar)
            {
                return VectorOp<AddAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S>
            SKT_FUNC SelfType& operator-=(S scalar)
            {
                return VectorOp<SubAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S>
            SKT_FUNC SelfType& operator*=(S scalar)
            {
                return VectorOp<MulAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S>
            SKT_FUNC SelfType& operator/=(S scalar)
            {
                return VectorOp<DivAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator%=(S scalar)
            {
                return VectorOp<ModAssignOp>::Apply(derived(), scalar);
            }

            // 位运算 - 向量
            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator&=(Vec<N, U> rhs)
            {
                return VectorOp<AndAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator|=(Vec<N, U> rhs)
            {
                return VectorOp<OrAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator^=(Vec<N, U> rhs)
            {
                return VectorOp<XorAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator<<=(Vec<N, U> rhs)
            {
                return VectorOp<LShiftAssignOp>::Apply(derived(), rhs);
            }

            template <typename U> requires CompatibleVectors<SelfType, Vec<N, U>> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator>>=(Vec<N, U> rhs)
            {
                return VectorOp<RShiftAssignOp>::Apply(derived(), rhs);
            }

            // 位运算 - 标量
            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator&=(S scalar)
            {
                return VectorOp<AndAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator|=(S scalar)
            {
                return VectorOp<OrAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator^=(S scalar)
            {
                return VectorOp<XorAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator<<=(S scalar)
            {
                return VectorOp<LShiftAssignOp>::Apply(derived(), scalar);
            }

            template <typename S> requires VectorScalarCompatible<SelfType, S> && IntVectorType<SelfType>
            SKT_FUNC SelfType& operator>>=(S scalar)
            {
                return VectorOp<RShiftAssignOp>::Apply(derived(), scalar);
            }
        };

    } // namespace Detail

} // namespace SKT

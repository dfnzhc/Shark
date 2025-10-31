/**
 * @File Concepts.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/10/31
 * @Brief This file is part of Shark.
 */

#pragma once

#include <concepts>
#include "Defines.hpp"

namespace SKT
{
    // clang-format off
    template <typename T> concept BoolType = std::is_same_v<bool, T>;
    template <typename T> concept U32Type  = std::is_same_v<u32, T>;
    template <typename T> concept U64Type  = std::is_same_v<u64, T>;
    template <typename T> concept F32Type  = std::is_same_v<f32, T>;
    template <typename T> concept F64Type  = std::is_same_v<f64, T>;

    template <typename T> concept SignedType   = std::is_signed_v<T>;
    template <typename T> concept UnsignedType = std::is_unsigned_v<T>;
    template <typename T> concept IntegralType = std::is_integral_v<T>;
    template <typename T> concept FloatType    = std::is_floating_point_v<T>;
    template <typename T> concept ArithType    = std::is_arithmetic_v<T>;
    // clang-format on

    template <ArithType A, ArithType B>
    using CommonType = std::common_type_t<A, B>;

    template <ArithType T>
    using MapFloatType = std::conditional_t<sizeof(T) <= 4, f32, f64>;

    template <typename T, typename... Ts>
    using CommonFloatType = MapFloatType<std::common_type_t<T, Ts...>>;

    template <typename T, typename U>
    concept BitwiseCompatible = IntegralType<T> && IntegralType<U>;

    // 用于数值类型转换
    template <ArithType T>
    constexpr T As(ArithType auto f) noexcept
        requires std::is_nothrow_convertible_v<decltype(f), T>
    {
        return static_cast<T>(f);
    }

    template <typename From, typename To>
    concept ConvertibleTo = std::is_convertible_v<From, To>;

    template <typename T, typename... Ts>
    concept AllConvertibleTo = (ConvertibleTo<T, Ts> && ...);


} // namespace SKT

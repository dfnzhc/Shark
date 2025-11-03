/**
 * @File VectorType.hpp
 * @Author dfnzhc (https://github.com/dfnzhc)
 * @Date 2025/11/3
 * @Brief This file is part of Shark.
 */

#pragma once

#include "VectorCore.hpp"

namespace SKT
{
    // ==================== 1D向量特化 ====================

    template <typename T>
    struct Vec<1, T> : public Detail::VectorBase<1, T, Vec<1, T>>
    {
        using Base                     = Detail::VectorBase<1, T, Vec<1, T>>;
        using ValueType                = T;
        using SelfType                 = Vec<1, T>;
        using BooleanType              = Vec<1, bool>;
        static constexpr int Dimension = 1;
        
        // clang-format off
        T x = {};

        // -------
        constexpr Vec()           = default;
        constexpr Vec(const Vec&) = default;
        
        explicit constexpr Vec(T value) noexcept : x(value) {}

        template <ArithType U> explicit constexpr Vec(U x_)      noexcept : x(As<T>(x_)) {}
        template <ArithType U> explicit constexpr Vec(Vec1<U> v) noexcept : x(As<T>(v.x)) {}
        template <ArithType U> explicit constexpr Vec(Vec2<U> v) noexcept : x(As<T>(v.x)) {}
        template <ArithType U> explicit constexpr Vec(Vec3<U> v) noexcept : x(As<T>(v.x)) {}
        template <ArithType U> explicit constexpr Vec(Vec4<U> v) noexcept : x(As<T>(v.x)) {}

        // -------
        constexpr Vec& operator=(const Vec&) noexcept = default;

        template <ArithType U>
        constexpr Vec& operator=(const Vec<1, U>& other) noexcept
        {
            x = As<T>(other.x);
            return *this;
        }

        // -------
        constexpr       T& operator[](int)       noexcept { return x; }
        constexpr const T& operator[](int) const noexcept { return x; }

        constexpr void set(T value) noexcept { x = value; }
        
        constexpr T value() const noexcept { return x; }
        explicit constexpr operator T() const noexcept { return x; }

        // -------
        static constexpr Vec Zero() noexcept { return Vec(T{}); }
        static constexpr Vec One() noexcept { return Vec(T{1}); }

        template <typename U = T> requires FloatType<U>
        static constexpr Vec UnitX() noexcept { return Vec(T{1}); }
        
        template <typename U = T> requires FloatType<U>
        static constexpr Vec Unit() noexcept { return Vec(T{1}); }
        // clang-format on

        // -------
        using Base::operator+=;
        using Base::operator-=;
        using Base::operator*=;
        using Base::operator/=;
        using Base::operator%=;
        using Base::operator&=;
        using Base::operator|=;
        using Base::operator^=;
        using Base::operator<<=;
        using Base::operator>>=;
    };

    // ==================== 2D向量特化 ====================

    template <typename T>
    struct Vec<2, T> : public Detail::VectorBase<2, T, Vec<2, T>>
    {
        using Base                     = Detail::VectorBase<2, T, Vec<2, T>>;
        using ValueType                = T;
        using SelfType                 = Vec<2, T>;
        using BooleanType              = Vec<2, bool>;
        static constexpr int Dimension = 2;

        T x = {}, y = {};
        // clang-format off

        // -------
        constexpr Vec()             noexcept = default;
        constexpr Vec(const Vec& v) noexcept = default;
    
        explicit constexpr Vec(T value) noexcept : x(value), y(value) {}
        constexpr Vec(T x_, T y_) noexcept : x(x_), y(y_) {}
        
        template <ArithType A, ArithType B> constexpr Vec(A x_, B y_) noexcept : x(As<T>(x_)), y(As<T>(y_)) {}
        template <ArithType A, ArithType B> constexpr Vec(Vec1<A> x_, B y_) noexcept : x(As<T>(x_.x)), y(As<T>(y_)) {}
        template <ArithType U> explicit constexpr Vec(Vec1<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.x)) {}
        template <ArithType U> explicit constexpr Vec(Vec2<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)) {}
        template <ArithType U> explicit constexpr Vec(Vec3<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)) {}
        template <ArithType U> explicit constexpr Vec(Vec4<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)) {}
    
        // -------
        constexpr Vec& operator=(const Vec&) noexcept = default;
    
        template<ArithType U>
        constexpr Vec& operator=(const Vec<2, U>& other) noexcept
        {
            x = As<T>(other.x);
            y = As<T>(other.y);
            return *this;
        }
    
        // -------
        constexpr       T& operator[](int index)       noexcept { return (index == 0) ? x : y;; }
        constexpr const T& operator[](int index) const noexcept { return (index == 0) ? x : y;; }
        
        constexpr void set(T x_, T y_) noexcept { x = x_; y = y_; }
    
        // -------
        static constexpr Vec Zero() noexcept { return Vec(T{}, T{}); }
        static constexpr Vec One() noexcept { return Vec(T{1}, T{1}); }
    
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitX() noexcept { return Vec(T{1}, T{}); }
    
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitY() noexcept { return Vec(T{}, T{1}); }

        template<typename U = T> requires FloatType<U>
        static constexpr Vec Unit() noexcept { return Vec(T{kInvSqrt2}, T{kInvSqrt2}); }

        // -------
        constexpr Vec<2, T> xx() const noexcept { return {x, x}; }
        constexpr Vec<2, T> xy() const noexcept { return {x, y}; }
        constexpr Vec<2, T> yx() const noexcept { return {y, x}; }
        constexpr Vec<2, T> yy() const noexcept { return {y, y}; }
    
        constexpr Vec<3, T> xxx() const noexcept { return {x, x, x}; }
        constexpr Vec<3, T> xxy() const noexcept { return {x, x, y}; }
        constexpr Vec<3, T> xyx() const noexcept { return {x, y, x}; }
        constexpr Vec<3, T> xyy() const noexcept { return {x, y, y}; }
        constexpr Vec<3, T> yxx() const noexcept { return {y, x, x}; }
        constexpr Vec<3, T> yxy() const noexcept { return {y, x, y}; }
        constexpr Vec<3, T> yyx() const noexcept { return {y, y, x}; }
        constexpr Vec<3, T> yyy() const noexcept { return {y, y, y}; }
    
        constexpr Vec<4, T> xxxx() const noexcept { return {x, x, x, x}; }
        constexpr Vec<4, T> xxxy() const noexcept { return {x, x, x, y}; }
        constexpr Vec<4, T> xxyx() const noexcept { return {x, x, y, x}; }
        constexpr Vec<4, T> xxyy() const noexcept { return {x, x, y, y}; }
        constexpr Vec<4, T> xyxx() const noexcept { return {x, y, x, x}; }
        constexpr Vec<4, T> xyxy() const noexcept { return {x, y, x, y}; }
        constexpr Vec<4, T> xyyx() const noexcept { return {x, y, y, x}; }
        constexpr Vec<4, T> xyyy() const noexcept { return {x, y, y, y}; }
        constexpr Vec<4, T> yxxx() const noexcept { return {y, x, x, x}; }
        constexpr Vec<4, T> yxxy() const noexcept { return {y, x, x, y}; }
        constexpr Vec<4, T> yxyx() const noexcept { return {y, x, y, x}; }
        constexpr Vec<4, T> yxyy() const noexcept { return {y, x, y, y}; }
        constexpr Vec<4, T> yyxx() const noexcept { return {y, y, x, x}; }
        constexpr Vec<4, T> yyxy() const noexcept { return {y, y, x, y}; }
        constexpr Vec<4, T> yyyx() const noexcept { return {y, y, y, x}; }
        constexpr Vec<4, T> yyyy() const noexcept { return {y, y, y, y}; }
        // clang-format on

        // -------
        using Base::operator+=;
        using Base::operator-=;
        using Base::operator*=;
        using Base::operator/=;
        using Base::operator%=;
        using Base::operator&=;
        using Base::operator|=;
        using Base::operator^=;
        using Base::operator<<=;
        using Base::operator>>=;
    };

    // ==================== 3D向量特化 ====================

    template <typename T>
    struct Vec<3, T> : public Detail::VectorBase<3, T, Vec<3, T>>
    {
        using Base                     = Detail::VectorBase<3, T, Vec<3, T>>;
        using ValueType                = T;
        using SelfType                 = Vec<3, T>;
        using BooleanType              = Vec<3, bool>;
        static constexpr int Dimension = 3;

        T x = {}, y = {}, z = {};
        // clang-format off
        
        // -------
        constexpr Vec()           noexcept = default;
        constexpr Vec(const Vec&) noexcept = default;
        
        explicit constexpr Vec(T value) noexcept : x(value), y(value), z(value) {}
        constexpr Vec(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
        
        template <ArithType X, ArithType Y, ArithType Z> constexpr Vec(X x_, Y y_, Z z_) noexcept : x(As<T>(x_)), y(As<T>(y_)), z(As<T>(z_)) {}
        template <ArithType X, ArithType Y, ArithType Z> constexpr Vec(Vec1<X> x_, Y y_, Z z_) noexcept : x(As<T>(x_.x)), y(As<T>(y_)), z(As<T>(z_)) {}
        template <ArithType A, ArithType B> constexpr Vec(Vec2<A> xy_, B z_) noexcept : x(As<T>(xy_.x)), y(As<T>(xy_.y)), z(As<T>(z_)) {}
        template <ArithType A, ArithType B> constexpr Vec(A x_, Vec2<B> yz_) noexcept : x(As<T>(x_)), y(As<T>(yz_.x)), z(As<T>(yz_.y)) {}
        template <ArithType U> explicit constexpr Vec(Vec1<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.x)), z(As<T>(v.x))  {}
        template <ArithType U> explicit constexpr Vec(Vec2<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z({})  {}
        template <ArithType U> explicit constexpr Vec(Vec3<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z(As<T>(v.z)) {}
        template <ArithType U> explicit constexpr Vec(Vec4<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z(As<T>(v.z)) {}
        
        // -------
        constexpr Vec& operator=(const Vec& other) noexcept = default;
        
        template<ArithType U>
        constexpr Vec& operator=(const Vec<3, U>& other) noexcept
        {
            x = As<T>(other.x);
            y = As<T>(other.y);
            z = As<T>(other.z);
            return *this;
        }
        
        // -------
        constexpr       T& operator[](int index)       noexcept { return (index == 0) ? x : (index == 1) ? y : z; }
        constexpr const T& operator[](int index) const noexcept { return (index == 0) ? x : (index == 1) ? y : z; }

        constexpr void set(T x_, T y_, T z_) noexcept { x = x_; y = y_; z = z_; }
        
        // -------
        static constexpr Vec Zero() noexcept { return Vec(T{}, T{}, T{}); }
        static constexpr Vec One() noexcept { return Vec(T{1}, T{1}, T{1}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitX() noexcept { return Vec(T{1}, T{}, T{}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitY() noexcept { return Vec(T{}, T{1}, T{}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitZ() noexcept { return Vec(T{}, T{}, T{1}); }

        template<typename U = T> requires FloatType<U>
        static constexpr Vec Unit() noexcept { return Vec(T{kInvSqrt3}, T{kInvSqrt3}, T{kInvSqrt3}); }
        
        // -------
        constexpr Vec<2, T> xx() const noexcept { return {x, x}; }
        constexpr Vec<2, T> xy() const noexcept { return {x, y}; }
        constexpr Vec<2, T> xz() const noexcept { return {x, z}; }
        constexpr Vec<2, T> yx() const noexcept { return {y, x}; }
        constexpr Vec<2, T> yy() const noexcept { return {y, y}; }
        constexpr Vec<2, T> yz() const noexcept { return {y, z}; }
        constexpr Vec<2, T> zx() const noexcept { return {z, x}; }
        constexpr Vec<2, T> zy() const noexcept { return {z, y}; }
        constexpr Vec<2, T> zz() const noexcept { return {z, z}; }
        
        constexpr Vec xxx() const noexcept { return {x, x, x}; }
        constexpr Vec xxy() const noexcept { return {x, x, y}; }
        constexpr Vec xxz() const noexcept { return {x, x, z}; }
        constexpr Vec xyx() const noexcept { return {x, y, x}; }
        constexpr Vec xyy() const noexcept { return {x, y, y}; }
        constexpr Vec xyz() const noexcept { return {x, y, z}; }
        constexpr Vec xzx() const noexcept { return {x, z, x}; }
        constexpr Vec xzy() const noexcept { return {x, z, y}; }
        constexpr Vec xzz() const noexcept { return {x, z, z}; }
        constexpr Vec yxx() const noexcept { return {y, x, x}; }
        constexpr Vec yxy() const noexcept { return {y, x, y}; }
        constexpr Vec yxz() const noexcept { return {y, x, z}; }
        constexpr Vec yyx() const noexcept { return {y, y, x}; }
        constexpr Vec yyy() const noexcept { return {y, y, y}; }
        constexpr Vec yyz() const noexcept { return {y, y, z}; }
        constexpr Vec yzx() const noexcept { return {y, z, x}; }
        constexpr Vec yzy() const noexcept { return {y, z, y}; }
        constexpr Vec yzz() const noexcept { return {y, z, z}; }
        constexpr Vec zxx() const noexcept { return {z, x, x}; }
        constexpr Vec zxy() const noexcept { return {z, x, y}; }
        constexpr Vec zxz() const noexcept { return {z, x, z}; }
        constexpr Vec zyx() const noexcept { return {z, y, x}; }
        constexpr Vec zyy() const noexcept { return {z, y, y}; }
        constexpr Vec zyz() const noexcept { return {z, y, z}; }
        constexpr Vec zzx() const noexcept { return {z, z, x}; }
        constexpr Vec zzy() const noexcept { return {z, z, y}; }
        constexpr Vec zzz() const noexcept { return {z, z, z}; }

        // -------
        using Base::operator+=;
        using Base::operator-=;
        using Base::operator*=;
        using Base::operator/=;
        using Base::operator%=;
        using Base::operator&=;
        using Base::operator|=;
        using Base::operator^=;
        using Base::operator<<=;
        using Base::operator>>=;
        // clang-format on
    };

    // ==================== 4D向量特化 ====================

    template <typename T>
    struct Vec<4, T> : public Detail::VectorBase<4, T, Vec<4, T>>
    {
        using Base                     = Detail::VectorBase<4, T, Vec<4, T>>;
        using ValueType                = T;
        using SelfType                 = Vec<4, T>;
        using BooleanType              = Vec<4, bool>;
        static constexpr int Dimension = 4;
        
        T x = {}, y = {}, z = {}, w = {};
        // clang-format off
        
        // -------
        constexpr Vec()                 noexcept = default;
        constexpr Vec(const Vec& other) noexcept = default;
        
        explicit constexpr Vec(T value) noexcept : x(value), y(value), z(value), w(value) {}
        constexpr Vec(T x_, T y_, T z_, T w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}

        template <ArithType X, ArithType Y, ArithType Z, ArithType W> constexpr Vec(X x_, Y y_, Z z_, W w_) noexcept : x(As<T>(x_)), y(As<T>(y_)), z(As<T>(z_)), w(As<T>(w_)) {}
        template <ArithType X, ArithType Y, ArithType Z, ArithType W> constexpr Vec(Vec1<X> x_, Y y_, Z z_, W w_) noexcept : x(As<T>(x_.x)), y(As<T>(y_)), z(As<T>(z_)), w(As<T>(w_)) {}
        template <ArithType A, ArithType B, ArithType C> constexpr Vec(Vec2<A> xy_, B z_, C w_) noexcept : x(As<T>(xy_.x)), y(As<T>(xy_.y)), z(As<T>(z_)), w(As<T>(w_)) {}
        template <ArithType A, ArithType B, ArithType C> constexpr Vec(A x_, Vec2<B> yz_, C w_) noexcept : x(As<T>(x_)), y(As<T>(yz_.x)), z(As<T>(yz_.y)), w(As<T>(w_)) {}
        template <ArithType A, ArithType B, ArithType C> constexpr Vec(A x_, B y_, Vec2<C> zw_) noexcept : x(As<T>(x_)), y(As<T>(y_)), z(As<T>(zw_.x)), w(As<T>(zw_.y)) {}
        template <ArithType A, ArithType B> constexpr Vec(Vec3<A> xyz_, B w_) noexcept : x(As<T>(xyz_.x)), y(As<T>(xyz_.y)), z(As<T>(xyz_.z)), w(As<T>(w_)) {}
        template <ArithType A, ArithType B> constexpr Vec(A x_, Vec3<B> yzw_) noexcept : x(As<T>(x_)), y(As<T>(yzw_.x)), z(As<T>(yzw_.y)), w(As<T>(yzw_.z)) {}
        template <ArithType A, ArithType B> constexpr Vec(Vec2<A> xy_, Vec2<B> zw_) noexcept : x(As<T>(xy_.x)), y(As<T>(xy_.y)), z(As<T>(zw_.x)), w(As<T>(zw_.y)) {}
        template <ArithType U> explicit constexpr Vec(Vec1<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.x)), z(As<T>(v.x)), w(As<T>(v.x))  {}
        template <ArithType U> explicit constexpr Vec(Vec2<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z({}), w({})  {}
        template <ArithType U> explicit constexpr Vec(Vec3<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z(As<T>(v.z)), w({}) {}
        template <ArithType U> explicit constexpr Vec(Vec4<U> v) noexcept : x(As<T>(v.x)), y(As<T>(v.y)), z(As<T>(v.z)), w(As<T>(v.w)) {}
        
        // -------
        constexpr Vec& operator=(const Vec& other) noexcept = default;
        
        template<ArithType U>
        constexpr Vec& operator=(const Vec<4, U>& other) noexcept
        {
            x = As<T>(other.x);
            y = As<T>(other.y);
            z = As<T>(other.z);
            w = As<T>(other.w);
            return *this;
        }
        
        // -------
        constexpr       T& operator[](int index)       noexcept { return (index == 0) ? x : (index == 1) ? y : (index == 2) ? z : w; }
        constexpr const T& operator[](int index) const noexcept { return (index == 0) ? x : (index == 1) ? y : (index == 2) ? z : w; }
        
        constexpr void set(T x_, T y_, T z_, T w_) noexcept { x = x_; y = y_; z = z_; w = w_; }
        
        // -------
        static constexpr Vec Zero() noexcept { return Vec(T{}, T{}, T{}, T{}); }
        static constexpr Vec One() noexcept { return Vec(T{1}, T{1}, T{1}, T{1}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitX() noexcept { return Vec(T{1}, T{}, T{}, T{}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitY() noexcept { return Vec(T{}, T{1}, T{}, T{}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitZ() noexcept { return Vec(T{}, T{}, T{1}, T{}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec UnitW() noexcept { return Vec(T{}, T{}, T{}, T{1}); }
        
        template<typename U = T> requires FloatType<U>
        static constexpr Vec Unit() noexcept { return Vec(T{0.5}, T{0.5}, T{0.5}, T{0.5}); }

        // -------
        constexpr Vec<2, T> xx() const noexcept { return {x, x}; }
        constexpr Vec<2, T> xy() const noexcept { return {x, y}; }
        constexpr Vec<2, T> xz() const noexcept { return {x, z}; }
        constexpr Vec<2, T> xw() const noexcept { return {x, w}; }
        constexpr Vec<2, T> yx() const noexcept { return {y, x}; }
        constexpr Vec<2, T> yy() const noexcept { return {y, y}; }
        constexpr Vec<2, T> yz() const noexcept { return {y, z}; }
        constexpr Vec<2, T> yw() const noexcept { return {y, w}; }
        constexpr Vec<2, T> zx() const noexcept { return {z, x}; }
        constexpr Vec<2, T> zy() const noexcept { return {z, y}; }
        constexpr Vec<2, T> zz() const noexcept { return {z, z}; }
        constexpr Vec<2, T> zw() const noexcept { return {z, w}; }
        constexpr Vec<2, T> wx() const noexcept { return {w, x}; }
        constexpr Vec<2, T> wy() const noexcept { return {w, y}; }
        constexpr Vec<2, T> wz() const noexcept { return {w, z}; }
        constexpr Vec<2, T> ww() const noexcept { return {w, w}; }
        
        constexpr Vec<3, T> xxx() const noexcept { return {x, x, x}; }
        constexpr Vec<3, T> xxy() const noexcept { return {x, x, y}; }
        constexpr Vec<3, T> xxz() const noexcept { return {x, x, z}; }
        constexpr Vec<3, T> xxw() const noexcept { return {x, x, w}; }
        constexpr Vec<3, T> xyx() const noexcept { return {x, y, x}; }
        constexpr Vec<3, T> xyy() const noexcept { return {x, y, y}; }
        constexpr Vec<3, T> xyz() const noexcept { return {x, y, z}; }
        constexpr Vec<3, T> xyw() const noexcept { return {x, y, w}; }
        constexpr Vec<3, T> xzx() const noexcept { return {x, z, x}; }
        constexpr Vec<3, T> xzy() const noexcept { return {x, z, y}; }
        constexpr Vec<3, T> xzz() const noexcept { return {x, z, z}; }
        constexpr Vec<3, T> xzw() const noexcept { return {x, z, w}; }
        constexpr Vec<3, T> xwx() const noexcept { return {x, w, x}; }
        constexpr Vec<3, T> xwy() const noexcept { return {x, w, y}; }
        constexpr Vec<3, T> xwz() const noexcept { return {x, w, z}; }
        constexpr Vec<3, T> xww() const noexcept { return {x, w, w}; }
        constexpr Vec<3, T> yxx() const noexcept { return {y, x, x}; }
        constexpr Vec<3, T> yxy() const noexcept { return {y, x, y}; }
        constexpr Vec<3, T> yxz() const noexcept { return {y, x, z}; }
        constexpr Vec<3, T> yxw() const noexcept { return {y, x, w}; }
        constexpr Vec<3, T> yyx() const noexcept { return {y, y, x}; }
        constexpr Vec<3, T> yyy() const noexcept { return {y, y, y}; }
        constexpr Vec<3, T> yyz() const noexcept { return {y, y, z}; }
        constexpr Vec<3, T> yyw() const noexcept { return {y, y, w}; }
        constexpr Vec<3, T> yzx() const noexcept { return {y, z, x}; }
        constexpr Vec<3, T> yzy() const noexcept { return {y, z, y}; }
        constexpr Vec<3, T> yzz() const noexcept { return {y, z, z}; }
        constexpr Vec<3, T> yzw() const noexcept { return {y, z, w}; }
        constexpr Vec<3, T> ywx() const noexcept { return {y, w, x}; }
        constexpr Vec<3, T> ywy() const noexcept { return {y, w, y}; }
        constexpr Vec<3, T> ywz() const noexcept { return {y, w, z}; }
        constexpr Vec<3, T> yww() const noexcept { return {y, w, w}; }
        constexpr Vec<3, T> zxx() const noexcept { return {z, x, x}; }
        constexpr Vec<3, T> zxy() const noexcept { return {z, x, y}; }
        constexpr Vec<3, T> zxz() const noexcept { return {z, x, z}; }
        constexpr Vec<3, T> zxw() const noexcept { return {z, x, w}; }
        constexpr Vec<3, T> zyx() const noexcept { return {z, y, x}; }
        constexpr Vec<3, T> zyy() const noexcept { return {z, y, y}; }
        constexpr Vec<3, T> zyz() const noexcept { return {z, y, z}; }
        constexpr Vec<3, T> zyw() const noexcept { return {z, y, w}; }
        constexpr Vec<3, T> zzx() const noexcept { return {z, z, x}; }
        constexpr Vec<3, T> zzy() const noexcept { return {z, z, y}; }
        constexpr Vec<3, T> zzz() const noexcept { return {z, z, z}; }
        constexpr Vec<3, T> zzw() const noexcept { return {z, z, w}; }
        constexpr Vec<3, T> zwx() const noexcept { return {z, w, x}; }
        constexpr Vec<3, T> zwy() const noexcept { return {z, w, y}; }
        constexpr Vec<3, T> zwz() const noexcept { return {z, w, z}; }
        constexpr Vec<3, T> zww() const noexcept { return {z, w, w}; }
        constexpr Vec<3, T> wxx() const noexcept { return {w, x, x}; }
        constexpr Vec<3, T> wxy() const noexcept { return {w, x, y}; }
        constexpr Vec<3, T> wxz() const noexcept { return {w, x, z}; }
        constexpr Vec<3, T> wxw() const noexcept { return {w, x, w}; }
        constexpr Vec<3, T> wyx() const noexcept { return {w, y, x}; }
        constexpr Vec<3, T> wyy() const noexcept { return {w, y, y}; }
        constexpr Vec<3, T> wyz() const noexcept { return {w, y, z}; }
        constexpr Vec<3, T> wyw() const noexcept { return {w, y, w}; }
        constexpr Vec<3, T> wzx() const noexcept { return {w, z, x}; }
        constexpr Vec<3, T> wzy() const noexcept { return {w, z, y}; }
        constexpr Vec<3, T> wzz() const noexcept { return {w, z, z}; }
        constexpr Vec<3, T> wzw() const noexcept { return {w, z, w}; }
        constexpr Vec<3, T> wwx() const noexcept { return {w, w, x}; }
        constexpr Vec<3, T> wwy() const noexcept { return {w, w, y}; }
        constexpr Vec<3, T> wwz() const noexcept { return {w, w, z}; }
        constexpr Vec<3, T> www() const noexcept { return {w, w, w}; }
        
        constexpr Vec xxxx() const noexcept { return {x, x, x, x}; }
        constexpr Vec xxxy() const noexcept { return {x, x, x, y}; }
        constexpr Vec xxxz() const noexcept { return {x, x, x, z}; }
        constexpr Vec xxxw() const noexcept { return {x, x, x, w}; }
        constexpr Vec xxyx() const noexcept { return {x, x, y, x}; }
        constexpr Vec xxyy() const noexcept { return {x, x, y, y}; }
        constexpr Vec xxyz() const noexcept { return {x, x, y, z}; }
        constexpr Vec xxyw() const noexcept { return {x, x, y, w}; }
        constexpr Vec xxzx() const noexcept { return {x, x, z, x}; }
        constexpr Vec xxzy() const noexcept { return {x, x, z, y}; }
        constexpr Vec xxzz() const noexcept { return {x, x, z, z}; }
        constexpr Vec xxzw() const noexcept { return {x, x, z, w}; }
        constexpr Vec xxwx() const noexcept { return {x, x, w, x}; }
        constexpr Vec xxwy() const noexcept { return {x, x, w, y}; }
        constexpr Vec xxwz() const noexcept { return {x, x, w, z}; }
        constexpr Vec xxww() const noexcept { return {x, x, w, w}; }
        constexpr Vec xyxx() const noexcept { return {x, y, x, x}; }
        constexpr Vec xyxy() const noexcept { return {x, y, x, y}; }
        constexpr Vec xyxz() const noexcept { return {x, y, x, z}; }
        constexpr Vec xyxw() const noexcept { return {x, y, x, w}; }
        constexpr Vec xyyx() const noexcept { return {x, y, y, x}; }
        constexpr Vec xyyy() const noexcept { return {x, y, y, y}; }
        constexpr Vec xyyz() const noexcept { return {x, y, y, z}; }
        constexpr Vec xyyw() const noexcept { return {x, y, y, w}; }
        constexpr Vec xyzx() const noexcept { return {x, y, z, x}; }
        constexpr Vec xyzy() const noexcept { return {x, y, z, y}; }
        constexpr Vec xyzz() const noexcept { return {x, y, z, z}; }
        constexpr Vec xyzw() const noexcept { return {x, y, z, w}; }
        constexpr Vec xywx() const noexcept { return {x, y, w, x}; }
        constexpr Vec xywy() const noexcept { return {x, y, w, y}; }
        constexpr Vec xywz() const noexcept { return {x, y, w, z}; }
        constexpr Vec xyww() const noexcept { return {x, y, w, w}; }
        constexpr Vec xzxx() const noexcept { return {x, z, x, x}; }
        constexpr Vec xzxy() const noexcept { return {x, z, x, y}; }
        constexpr Vec xzxz() const noexcept { return {x, z, x, z}; }
        constexpr Vec xzxw() const noexcept { return {x, z, x, w}; }
        constexpr Vec xzyx() const noexcept { return {x, z, y, x}; }
        constexpr Vec xzyy() const noexcept { return {x, z, y, y}; }
        constexpr Vec xzyz() const noexcept { return {x, z, y, z}; }
        constexpr Vec xzyw() const noexcept { return {x, z, y, w}; }
        constexpr Vec xzzx() const noexcept { return {x, z, z, x}; }
        constexpr Vec xzzy() const noexcept { return {x, z, z, y}; }
        constexpr Vec xzzz() const noexcept { return {x, z, z, z}; }
        constexpr Vec xzzw() const noexcept { return {x, z, z, w}; }
        constexpr Vec xzwx() const noexcept { return {x, z, w, x}; }
        constexpr Vec xzwy() const noexcept { return {x, z, w, y}; }
        constexpr Vec xzwz() const noexcept { return {x, z, w, z}; }
        constexpr Vec xzww() const noexcept { return {x, z, w, w}; }
        constexpr Vec xwxx() const noexcept { return {x, w, x, x}; }
        constexpr Vec xwxy() const noexcept { return {x, w, x, y}; }
        constexpr Vec xwxz() const noexcept { return {x, w, x, z}; }
        constexpr Vec xwxw() const noexcept { return {x, w, x, w}; }
        constexpr Vec xwyx() const noexcept { return {x, w, y, x}; }
        constexpr Vec xwyy() const noexcept { return {x, w, y, y}; }
        constexpr Vec xwyz() const noexcept { return {x, w, y, z}; }
        constexpr Vec xwyw() const noexcept { return {x, w, y, w}; }
        constexpr Vec xwzx() const noexcept { return {x, w, z, x}; }
        constexpr Vec xwzy() const noexcept { return {x, w, z, y}; }
        constexpr Vec xwzz() const noexcept { return {x, w, z, z}; }
        constexpr Vec xwzw() const noexcept { return {x, w, z, w}; }
        constexpr Vec xwwx() const noexcept { return {x, w, w, x}; }
        constexpr Vec xwwy() const noexcept { return {x, w, w, y}; }
        constexpr Vec xwwz() const noexcept { return {x, w, w, z}; }
        constexpr Vec xwww() const noexcept { return {x, w, w, w}; }
        constexpr Vec yxxx() const noexcept { return {y, x, x, x}; }
        constexpr Vec yxxy() const noexcept { return {y, x, x, y}; }
        constexpr Vec yxxz() const noexcept { return {y, x, x, z}; }
        constexpr Vec yxxw() const noexcept { return {y, x, x, w}; }
        constexpr Vec yxyx() const noexcept { return {y, x, y, x}; }
        constexpr Vec yxyy() const noexcept { return {y, x, y, y}; }
        constexpr Vec yxyz() const noexcept { return {y, x, y, z}; }
        constexpr Vec yxyw() const noexcept { return {y, x, y, w}; }
        constexpr Vec yxzx() const noexcept { return {y, x, z, x}; }
        constexpr Vec yxzy() const noexcept { return {y, x, z, y}; }
        constexpr Vec yxzz() const noexcept { return {y, x, z, z}; }
        constexpr Vec yxzw() const noexcept { return {y, x, z, w}; }
        constexpr Vec yxwx() const noexcept { return {y, x, w, x}; }
        constexpr Vec yxwy() const noexcept { return {y, x, w, y}; }
        constexpr Vec yxwz() const noexcept { return {y, x, w, z}; }
        constexpr Vec yxww() const noexcept { return {y, x, w, w}; }
        constexpr Vec yyxx() const noexcept { return {y, y, x, x}; }
        constexpr Vec yyxy() const noexcept { return {y, y, x, y}; }
        constexpr Vec yyxz() const noexcept { return {y, y, x, z}; }
        constexpr Vec yyxw() const noexcept { return {y, y, x, w}; }
        constexpr Vec yyyx() const noexcept { return {y, y, y, x}; }
        constexpr Vec yyyy() const noexcept { return {y, y, y, y}; }
        constexpr Vec yyyz() const noexcept { return {y, y, y, z}; }
        constexpr Vec yyyw() const noexcept { return {y, y, y, w}; }
        constexpr Vec yyzx() const noexcept { return {y, y, z, x}; }
        constexpr Vec yyzy() const noexcept { return {y, y, z, y}; }
        constexpr Vec yyzz() const noexcept { return {y, y, z, z}; }
        constexpr Vec yyzw() const noexcept { return {y, y, z, w}; }
        constexpr Vec yywx() const noexcept { return {y, y, w, x}; }
        constexpr Vec yywy() const noexcept { return {y, y, w, y}; }
        constexpr Vec yywz() const noexcept { return {y, y, w, z}; }
        constexpr Vec yyww() const noexcept { return {y, y, w, w}; }
        constexpr Vec yzxx() const noexcept { return {y, z, x, x}; }
        constexpr Vec yzxy() const noexcept { return {y, z, x, y}; }
        constexpr Vec yzxz() const noexcept { return {y, z, x, z}; }
        constexpr Vec yzxw() const noexcept { return {y, z, x, w}; }
        constexpr Vec yzyx() const noexcept { return {y, z, y, x}; }
        constexpr Vec yzyy() const noexcept { return {y, z, y, y}; }
        constexpr Vec yzyz() const noexcept { return {y, z, y, z}; }
        constexpr Vec yzyw() const noexcept { return {y, z, y, w}; }
        constexpr Vec yzzx() const noexcept { return {y, z, z, x}; }
        constexpr Vec yzzy() const noexcept { return {y, z, z, y}; }
        constexpr Vec yzzz() const noexcept { return {y, z, z, z}; }
        constexpr Vec yzzw() const noexcept { return {y, z, z, w}; }
        constexpr Vec yzwx() const noexcept { return {y, z, w, x}; }
        constexpr Vec yzwy() const noexcept { return {y, z, w, y}; }
        constexpr Vec yzwz() const noexcept { return {y, z, w, z}; }
        constexpr Vec yzww() const noexcept { return {y, z, w, w}; }
        constexpr Vec ywxx() const noexcept { return {y, w, x, x}; }
        constexpr Vec ywxy() const noexcept { return {y, w, x, y}; }
        constexpr Vec ywxz() const noexcept { return {y, w, x, z}; }
        constexpr Vec ywxw() const noexcept { return {y, w, x, w}; }
        constexpr Vec ywyx() const noexcept { return {y, w, y, x}; }
        constexpr Vec ywyy() const noexcept { return {y, w, y, y}; }
        constexpr Vec ywyz() const noexcept { return {y, w, y, z}; }
        constexpr Vec ywyw() const noexcept { return {y, w, y, w}; }
        constexpr Vec ywzx() const noexcept { return {y, w, z, x}; }
        constexpr Vec ywzy() const noexcept { return {y, w, z, y}; }
        constexpr Vec ywzz() const noexcept { return {y, w, z, z}; }
        constexpr Vec ywzw() const noexcept { return {y, w, z, w}; }
        constexpr Vec ywwx() const noexcept { return {y, w, w, x}; }
        constexpr Vec ywwy() const noexcept { return {y, w, w, y}; }
        constexpr Vec ywwz() const noexcept { return {y, w, w, z}; }
        constexpr Vec ywww() const noexcept { return {y, w, w, w}; }
        constexpr Vec zxxx() const noexcept { return {z, x, x, x}; }
        constexpr Vec zxxy() const noexcept { return {z, x, x, y}; }
        constexpr Vec zxxz() const noexcept { return {z, x, x, z}; }
        constexpr Vec zxxw() const noexcept { return {z, x, x, w}; }
        constexpr Vec zxyx() const noexcept { return {z, x, y, x}; }
        constexpr Vec zxyy() const noexcept { return {z, x, y, y}; }
        constexpr Vec zxyz() const noexcept { return {z, x, y, z}; }
        constexpr Vec zxyw() const noexcept { return {z, x, y, w}; }
        constexpr Vec zxzx() const noexcept { return {z, x, z, x}; }
        constexpr Vec zxzy() const noexcept { return {z, x, z, y}; }
        constexpr Vec zxzz() const noexcept { return {z, x, z, z}; }
        constexpr Vec zxzw() const noexcept { return {z, x, z, w}; }
        constexpr Vec zxwx() const noexcept { return {z, x, w, x}; }
        constexpr Vec zxwy() const noexcept { return {z, x, w, y}; }
        constexpr Vec zxwz() const noexcept { return {z, x, w, z}; }
        constexpr Vec zxww() const noexcept { return {z, x, w, w}; }
        constexpr Vec zyxx() const noexcept { return {z, y, x, x}; }
        constexpr Vec zyxy() const noexcept { return {z, y, x, y}; }
        constexpr Vec zyxz() const noexcept { return {z, y, x, z}; }
        constexpr Vec zyxw() const noexcept { return {z, y, x, w}; }
        constexpr Vec zyyx() const noexcept { return {z, y, y, x}; }
        constexpr Vec zyyy() const noexcept { return {z, y, y, y}; }
        constexpr Vec zyyz() const noexcept { return {z, y, y, z}; }
        constexpr Vec zyyw() const noexcept { return {z, y, y, w}; }
        constexpr Vec zyzx() const noexcept { return {z, y, z, x}; }
        constexpr Vec zyzy() const noexcept { return {z, y, z, y}; }
        constexpr Vec zyzz() const noexcept { return {z, y, z, z}; }
        constexpr Vec zyzw() const noexcept { return {z, y, z, w}; }
        constexpr Vec zywx() const noexcept { return {z, y, w, x}; }
        constexpr Vec zywy() const noexcept { return {z, y, w, y}; }
        constexpr Vec zywz() const noexcept { return {z, y, w, z}; }
        constexpr Vec zyww() const noexcept { return {z, y, w, w}; }
        constexpr Vec zzxx() const noexcept { return {z, z, x, x}; }
        constexpr Vec zzxy() const noexcept { return {z, z, x, y}; }
        constexpr Vec zzxz() const noexcept { return {z, z, x, z}; }
        constexpr Vec zzxw() const noexcept { return {z, z, x, w}; }
        constexpr Vec zzyx() const noexcept { return {z, z, y, x}; }
        constexpr Vec zzyy() const noexcept { return {z, z, y, y}; }
        constexpr Vec zzyz() const noexcept { return {z, z, y, z}; }
        constexpr Vec zzyw() const noexcept { return {z, z, y, w}; }
        constexpr Vec zzzx() const noexcept { return {z, z, z, x}; }
        constexpr Vec zzzy() const noexcept { return {z, z, z, y}; }
        constexpr Vec zzzz() const noexcept { return {z, z, z, z}; }
        constexpr Vec zzzw() const noexcept { return {z, z, z, w}; }
        constexpr Vec zzwx() const noexcept { return {z, z, w, x}; }
        constexpr Vec zzwy() const noexcept { return {z, z, w, y}; }
        constexpr Vec zzwz() const noexcept { return {z, z, w, z}; }
        constexpr Vec zzww() const noexcept { return {z, z, w, w}; }
        constexpr Vec zwxx() const noexcept { return {z, w, x, x}; }
        constexpr Vec zwxy() const noexcept { return {z, w, x, y}; }
        constexpr Vec zwxz() const noexcept { return {z, w, x, z}; }
        constexpr Vec zwxw() const noexcept { return {z, w, x, w}; }
        constexpr Vec zwyx() const noexcept { return {z, w, y, x}; }
        constexpr Vec zwyy() const noexcept { return {z, w, y, y}; }
        constexpr Vec zwyz() const noexcept { return {z, w, y, z}; }
        constexpr Vec zwyw() const noexcept { return {z, w, y, w}; }
        constexpr Vec zwzx() const noexcept { return {z, w, z, x}; }
        constexpr Vec zwzy() const noexcept { return {z, w, z, y}; }
        constexpr Vec zwzz() const noexcept { return {z, w, z, z}; }
        constexpr Vec zwzw() const noexcept { return {z, w, z, w}; }
        constexpr Vec zwwx() const noexcept { return {z, w, w, x}; }
        constexpr Vec zwwy() const noexcept { return {z, w, w, y}; }
        constexpr Vec zwwz() const noexcept { return {z, w, w, z}; }
        constexpr Vec zwww() const noexcept { return {z, w, w, w}; }
        constexpr Vec wxxx() const noexcept { return {w, x, x, x}; }
        constexpr Vec wxxy() const noexcept { return {w, x, x, y}; }
        constexpr Vec wxxz() const noexcept { return {w, x, x, z}; }
        constexpr Vec wxxw() const noexcept { return {w, x, x, w}; }
        constexpr Vec wxyx() const noexcept { return {w, x, y, x}; }
        constexpr Vec wxyy() const noexcept { return {w, x, y, y}; }
        constexpr Vec wxyz() const noexcept { return {w, x, y, z}; }
        constexpr Vec wxyw() const noexcept { return {w, x, y, w}; }
        constexpr Vec wxzx() const noexcept { return {w, x, z, x}; }
        constexpr Vec wxzy() const noexcept { return {w, x, z, y}; }
        constexpr Vec wxzz() const noexcept { return {w, x, z, z}; }
        constexpr Vec wxzw() const noexcept { return {w, x, z, w}; }
        constexpr Vec wxwx() const noexcept { return {w, x, w, x}; }
        constexpr Vec wxwy() const noexcept { return {w, x, w, y}; }
        constexpr Vec wxwz() const noexcept { return {w, x, w, z}; }
        constexpr Vec wxww() const noexcept { return {w, x, w, w}; }
        constexpr Vec wyxx() const noexcept { return {w, y, x, x}; }
        constexpr Vec wyxy() const noexcept { return {w, y, x, y}; }
        constexpr Vec wyxz() const noexcept { return {w, y, x, z}; }
        constexpr Vec wyxw() const noexcept { return {w, y, x, w}; }
        constexpr Vec wyyx() const noexcept { return {w, y, y, x}; }
        constexpr Vec wyyy() const noexcept { return {w, y, y, y}; }
        constexpr Vec wyyz() const noexcept { return {w, y, y, z}; }
        constexpr Vec wyyw() const noexcept { return {w, y, y, w}; }
        constexpr Vec wyzx() const noexcept { return {w, y, z, x}; }
        constexpr Vec wyzy() const noexcept { return {w, y, z, y}; }
        constexpr Vec wyzz() const noexcept { return {w, y, z, z}; }
        constexpr Vec wyzw() const noexcept { return {w, y, z, w}; }
        constexpr Vec wywx() const noexcept { return {w, y, w, x}; }
        constexpr Vec wywy() const noexcept { return {w, y, w, y}; }
        constexpr Vec wywz() const noexcept { return {w, y, w, z}; }
        constexpr Vec wyww() const noexcept { return {w, y, w, w}; }
        constexpr Vec wzxx() const noexcept { return {w, z, x, x}; }
        constexpr Vec wzxy() const noexcept { return {w, z, x, y}; }
        constexpr Vec wzxz() const noexcept { return {w, z, x, z}; }
        constexpr Vec wzxw() const noexcept { return {w, z, x, w}; }
        constexpr Vec wzyx() const noexcept { return {w, z, y, x}; }
        constexpr Vec wzyy() const noexcept { return {w, z, y, y}; }
        constexpr Vec wzyz() const noexcept { return {w, z, y, z}; }
        constexpr Vec wzyw() const noexcept { return {w, z, y, w}; }
        constexpr Vec wzzx() const noexcept { return {w, z, z, x}; }
        constexpr Vec wzzy() const noexcept { return {w, z, z, y}; }
        constexpr Vec wzzz() const noexcept { return {w, z, z, z}; }
        constexpr Vec wzzw() const noexcept { return {w, z, z, w}; }
        constexpr Vec wzwx() const noexcept { return {w, z, w, x}; }
        constexpr Vec wzwy() const noexcept { return {w, z, w, y}; }
        constexpr Vec wzwz() const noexcept { return {w, z, w, z}; }
        constexpr Vec wzww() const noexcept { return {w, z, w, w}; }
        constexpr Vec wwxx() const noexcept { return {w, w, x, x}; }
        constexpr Vec wwxy() const noexcept { return {w, w, x, y}; }
        constexpr Vec wwxz() const noexcept { return {w, w, x, z}; }
        constexpr Vec wwxw() const noexcept { return {w, w, x, w}; }
        constexpr Vec wwyx() const noexcept { return {w, w, y, x}; }
        constexpr Vec wwyy() const noexcept { return {w, w, y, y}; }
        constexpr Vec wwyz() const noexcept { return {w, w, y, z}; }
        constexpr Vec wwyw() const noexcept { return {w, w, y, w}; }
        constexpr Vec wwzx() const noexcept { return {w, w, z, x}; }
        constexpr Vec wwzy() const noexcept { return {w, w, z, y}; }
        constexpr Vec wwzz() const noexcept { return {w, w, z, z}; }
        constexpr Vec wwzw() const noexcept { return {w, w, z, w}; }
        constexpr Vec wwwx() const noexcept { return {w, w, w, x}; }
        constexpr Vec wwwy() const noexcept { return {w, w, w, y}; }
        constexpr Vec wwwz() const noexcept { return {w, w, w, z}; }
        constexpr Vec wwww() const noexcept { return {w, w, w, w}; }

        // -------
        using Base::operator+=;
        using Base::operator-=;
        using Base::operator*=;
        using Base::operator/=;
        using Base::operator%=;
        using Base::operator&=;
        using Base::operator|=;
        using Base::operator^=;
        using Base::operator<<=;
        using Base::operator>>=;
        // clang-format on
    };

} // namespace SKT

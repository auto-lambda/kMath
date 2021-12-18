// Copyright© 2021, The Lambda Project <https://github.com/auto-lambda>
// All rights reserved.
//
// This file is part of The kMath Project which is made available under the BSD 3-Clause License.
// See LICENSE at the root directory or visit https://github.com/auto-lambda/kMath/blob/master/LICENSE
// for full license details.
//
// SPDX-License-Identifier: BSD-3-Clause
#ifndef KMATH_MATH_HPP_
#define KMATH_MATH_HPP_

#ifdef __cpp_constinit
# define KMATH_CXX20_CONSTINIT constinit
#else
# define KMATH_CXX20_CONSTINIT constexpr
#endif  // __cpp_constinit

namespace math::legal {
[[maybe_unused]] KMATH_CXX20_CONSTINIT auto kLicense = R"(The kMath library
is made available under the BSD 3-Clause License.

Copyright (c) 2021, The Lambda Project <https://github.com/auto-lambda>
All rights reserved.

See https://github.com/auto-lambda/kMath/blob/master/LICENSE for full license text.
SPDX-License-Identifier: BSD-3-Clause)";
#ifndef __cpp_constinit
[[maybe_unused]] struct Embed {
  char const *str = ::math::legal::kLicense;
} embed {};
#endif  // __cpp_constinit
}  // namespace math::legal

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <stdexcept>

#if __has_include(<iostream>) && defined(KMATH_IOSTREAM)
# include <iostream>
#endif  // __has_include(<iostream>) && !defined(KMATH_IOSTREAM)

#if __has_cpp_attribute(no_unique_address)
# define KMATH_CXX20_NO_UNIQUE_ADDR [[no_unique_address]]
#else
# define KMATH_CXX20_NO_UNIQUE_ADDR
#endif  // __has_cpp_attribute(no_unique_address)

#ifdef __cpp_consteval
# define KMATH_CXX20_CONSTEVAL consteval
#else
# define KMATH_CXX20_CONSTEVAL constexpr
#endif  // __cpp_consteval

#ifndef declauto
# define declauto decltype(auto)
#endif  // declauto

namespace math {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/*========================*/
/* Mathematical constants */
/*========================*/
/// @brief Euler's number constant
template <Arithmetic T>
inline constexpr T kEulersNumber{2.71828182845904523536L};
/// @brief log2(e) constant
template <Arithmetic T>
inline constexpr T kLog2e{1.44269504088896340736L};
/// @brief log10(e) constant
template <Arithmetic T>
inline constexpr T kLog10e{0.434294481903251827651L};
/// @brief ln(2) constant
template <Arithmetic T>
inline constexpr T kLn2{0.693147180559945309417L};
/// @brief ln(10) constant
template <Arithmetic T>
inline constexpr T kLn10{2.30258509299404568402L};
/// @brief π constant
template <Arithmetic T>
inline constexpr T kPi{3.14159265358979323846L};
/// @brief π/2 constant
template <Arithmetic T>
inline constexpr T kPiOver2{1.57079632679489661923L};
/// @brief π/4 constant
template <Arithmetic T>
inline constexpr T kPiOver4{0.785398163397448309616L};
/// @brief √2 constant
template <Arithmetic T>
inline constexpr T kSqrt2{1.41421356237309504880L};
/// @brief 1/π constant
template <Arithmetic T>
inline constexpr T kOneOverPi{0.318309886183790671538L};
/// @brief 2/π constant
template <Arithmetic T>
inline constexpr T kTwoOverPi{0.636619772367581343076L};
/// @brief 2/√π constant
template <Arithmetic T>
inline constexpr T kTwoOverSqrtPi{1.12837916709551257390L};
/// @brief 1/√2 constant
template <Arithmetic T>
inline constexpr T kTwoOverSqrt2{0.707106781186547524401L};
/// @brief quiet nan constant
template <Arithmetic T>
inline constexpr T kQuietNan = std::numeric_limits<T>::quiet_NaN();
/// @brief signal nan constant
template <Arithmetic T>
inline constexpr T kSignalingNan = std::numeric_limits<T>::signaling_NaN();
/// @brief nan constant
template <Arithmetic T>
inline constexpr T kNan = kQuietNan<T>;
/// @brief inf constant
template <Arithmetic T>
inline constexpr T kInf = std::numeric_limits<T>::infinity();
/// @brief epsilon constant
template <Arithmetic T>
inline constexpr T kEpsilon = std::numeric_limits<T>::epsilon();

/// @brief Type used to perform pointer arithmetic
using SizeType = std::array<std::nullptr_t, 1>::size_type;

template <Arithmetic T, SizeType Dims>
struct VectorStorage;

template <Arithmetic T, SizeType Dims>
struct Vector;

// clang-format off
namespace internal {
/// @brief Utility type to disable copy-assignment
struct NonCopyAssignable {
  auto &operator=(NonCopyAssignable const &) = delete;
  auto &operator=(NonCopyAssignable) = delete;
};

/// @brief Utility type to disable move-assignment
struct NonMoveAssignable {
  auto &operator=(NonMoveAssignable &&) = delete;
};

/// @brief Utility type to disable copy-construction
struct NonCopyConstructible {
  NonCopyConstructible(NonCopyConstructible const &) = delete;
};

/// @brief Utility type to disable move-construction
struct NonMoveConstructible {
  NonMoveConstructible(NonMoveConstructible &&) = delete;
};

/// @brief Utility type to disable default-construction
struct NonDefaultConstructible {
  NonDefaultConstructible() = delete;
};

/// @brief Non-constructible and non-assignable Tag type
struct Tag : NonDefaultConstructible, NonCopyConstructible, NonMoveConstructible
                                    , NonCopyAssignable   , NonMoveAssignable
{};

static_assert(!std::is_constructible_v<Tag>      && !std::is_assignable_v<Tag, Tag>
           && !std::is_copy_constructible_v<Tag> && !std::is_copy_assignable_v<Tag>
           && !std::is_move_constructible_v<Tag> && !std::is_move_assignable_v<Tag>);

/// @brief Tag used to identify a Vector
struct VectorTag     : Tag{};
/// @brief Tag used to identify a Matrix
struct MatrixTag     : Tag{};
/// @brief Tag used to identify a Quaternion
struct QuaternionTag : Tag{};

/// @brief Alias for std::remove_cvref_t<T>
template <typename T> using NoCvRef = std::remove_cvref_t<T>;
/// @brief Alias for std::remove_cv_t<T>
template <typename T> using NoCv    = std::remove_cv_t<T>;

template <bool ShouldCopy, typename T>
concept CopyOrNonConstLValue = ShouldCopy || !(std::is_lvalue_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>);
}  // namespace internal

/// @brief Determines whether type T is of Vector based on its Tag
template <typename T> concept AVector     = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::VectorTag>;
/// @brief Determines whether type T is of Matrix based on its Tag
template <typename T> concept AMatrix     = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::MatrixTag>;
/// @brief Determines whether type T is of Quaternion based on its Tag
template <typename T> concept AQuaternion = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::QuaternionTag>;

namespace internal {
[[nodiscard]] consteval std::size_t align(std::size_t const size) noexcept {
    constexpr auto kXmmAlignment = sizeof(std::uint64_t) * 2;
    constexpr auto kYmmAlignment = sizeof(std::uint64_t) * 4;
    constexpr auto kZmmAlignment = sizeof(std::uint64_t) * 8;

    if (size == 0) return 1;
    if (size <= sizeof(std::uint32_t)) return alignof(std::uint32_t);
    if (size <= sizeof(std::uint64_t)) return alignof(std::uint64_t);
    if (size <= kXmmAlignment) return kXmmAlignment;
    if (size <= kYmmAlignment) return kYmmAlignment;
    if (size <= kZmmAlignment) return kZmmAlignment;
    return kZmmAlignment;
}

template <Arithmetic T, std::size_t Dims = 0>
struct VectorStorage {
  constexpr VectorStorage() noexcept = default;
  explicit constexpr VectorStorage(std::array<T, Dims> const &data) noexcept
      : data_{data} {}

  template <Arithmetic... Args>
  constexpr VectorStorage(Args &&...args) noexcept
      : data_{std::forward<Args>(args)...} {}

  explicit constexpr VectorStorage(T const *raw) noexcept
      : data_{[]<std::size_t...Is>(T const *ptr,
                                   std::index_sequence<Is...>) constexpr {
            return std::array<T, sizeof...(Is)>{ptr[Is]...};
        }(raw, std::make_index_sequence<Dims>{})}
  {}
  
  [[nodiscard]] friend constexpr auto operator<=>(
      VectorStorage const &left, VectorStorage const &right) noexcept {
    return left.data_ <=> right.data_;
  }

  [[nodiscard]] friend constexpr auto operator==(
      VectorStorage const &left, VectorStorage const &right) noexcept {
    return left.data_ == right.data_;
  }

  using StorageType = typename std::array<T, Dims>;
  alignas(internal::align(sizeof(StorageType))) StorageType data_{};
};

template <Arithmetic T>
struct VectorStorage<T, 0> {
  KMATH_CXX20_NO_UNIQUE_ADDR struct Empty {} data_{};
};

[[nodiscard]] constexpr auto newton_raphson(long double scalar, long double cur, long double prev) noexcept {
  if (cur == prev)
    return cur;
  else
    return newton_raphson(scalar, static_cast<long double>(.5) * (cur + scalar / cur), cur);
}

[[nodiscard]] constexpr auto ct_sqrt(Arithmetic auto scalar) noexcept {
  using Scalar = internal::NoCvRef<decltype(scalar)>;
  return scalar >= kEpsilon<Scalar> && scalar < kInf<Scalar>
    ? static_cast<Scalar>(newton_raphson(scalar, scalar, Scalar{}))
    : kQuietNan<Scalar>;
}
// clang-format on
}  // namespace internal

[[nodiscard]] constexpr auto ct_sqrt(Arithmetic auto scalar) noexcept {
  return std::is_constant_evaluated()
    ? ::math::internal::ct_sqrt(scalar)
    : static_cast<internal::NoCvRef<decltype(scalar)>>(std::sqrt(scalar));
}

[[nodiscard]] constexpr auto dot(AVector auto &&lhs,
                                 AVector auto &&rhs) noexcept {
  static_assert(std::is_same_v<decltype(lhs.data_), decltype(rhs.data_)>,
                "Cannot perform dot() on vector types of different dimensions "
                "or scalar type.");
  using VectorType = typename internal::NoCvRef<decltype(lhs)>;
  typename VectorType::Scalar sum{};
  for (SizeType i{}; i < VectorType::kDims; ++i) sum += lhs[i] * rhs[i];
  return sum;
}

[[nodiscard]] constexpr auto cross(AVector auto &&lhs,
                                   AVector auto &&rhs) noexcept {
  // TODO: Implement multi-dimensional cross()
  using VectorType = typename internal::NoCvRef<decltype(lhs)>;
  static_assert(VectorType::kDims == internal::NoCvRef<decltype(rhs)>::kDims,
                "Cannot perform cross() on non 3-dimensional vectors.");
  return Vector<typename VectorType::Scalar, 3>{
      lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
      lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

namespace internal {
// clang-format off
template <bool ShouldCopy>
constexpr declauto implement_arithmetic(auto &&lhs, auto &&rhs, auto &&op) noexcept {
  using Lhs = NoCvRef<decltype(lhs)>;
  using Rhs = NoCvRef<decltype(rhs)>;

  constexpr auto cannot_throw = !ShouldCopy || (std::is_nothrow_copy_constructible_v<Lhs>
                                             && std::is_nothrow_copy_constructible_v<Rhs>);

  declauto result = [&]() constexpr noexcept(cannot_throw) -> declauto {
    if constexpr (AVector<Lhs> && AVector<Rhs>) {
      if constexpr (ShouldCopy)
        if constexpr (Lhs::kDims <= Rhs::kDims) return static_cast<Lhs>(lhs);
        else static_cast<Rhs>(rhs);
      else
        return Lhs::kDims <= Rhs::kDims ? lhs : rhs;
    } else if constexpr (AVector<Lhs>) {
      if constexpr (ShouldCopy) return static_cast<Lhs>(lhs);
      else return lhs;
    } else {
      if constexpr (ShouldCopy) return static_cast<Rhs>(rhs);
      else return rhs;
    }
  }();

  constexpr auto bind_op = [wrapper = [](auto &&arg) constexpr noexcept -> declauto {
    struct Wrapper {
      std::remove_reference_t<decltype(arg)> &scalar_ref;
      
      constexpr declauto operator[](SizeType const)       noexcept { return scalar_ref; }
      constexpr declauto operator[](SizeType const) const noexcept { return scalar_ref; }
    };
    return Wrapper{arg};
  }](auto &&arg) constexpr noexcept -> declauto {
    if constexpr (AVector<decltype(arg)>)
      return arg;
    else
      return wrapper(arg);
  };

  declauto op1 = bind_op(std::forward<decltype(lhs)>(lhs));
  declauto op2 = bind_op(std::forward<decltype(rhs)>(rhs));
  for (SizeType i{}; auto &&scalar : result) {
    scalar = op(op1[i], op2[i]);
    ++i;
  }

  return result;
}
}  // namespace internal

#define KMATH_IMPL_ARITHMETIC_IMPL(ShouldCopy, Token, Dims, Op)                                                  \
  template <typename U, typename V>                                                                              \
    requires (::math::AVector<U> || ::math::AVector<V>) && ::math::internal::CopyOrNonConstLValue<ShouldCopy, U> \
  constexpr declauto operator Token(U &&u, V &&v) noexcept {                                                     \
    return ::math::internal::implement_arithmetic<ShouldCopy>(std::forward<U>(u), std::forward<V>(v), Op<>{});   \
  }

// clang-format on

#define KMATH_IMPL_ARITHMETIC(Token, Dims, Op)          \
  KMATH_IMPL_ARITHMETIC_IMPL(true, Token, Dims, Op)     \
  KMATH_IMPL_ARITHMETIC_IMPL(false, Token##=, Dims, Op)

KMATH_IMPL_ARITHMETIC(-, Dims, std::minus)
KMATH_IMPL_ARITHMETIC(+, Dims, std::plus)
KMATH_IMPL_ARITHMETIC(/, Dims, std::divides)
KMATH_IMPL_ARITHMETIC(*, Dims, std::multiplies)

#undef KMATH_IMPL_ARITHMETIC
#undef KMATH_IMPL_ARITHMETIC_IMPL

/// @class Vector
/// @brief Provides an abstraction of mathematical vectors around Dims values of type T
/// 
/// @tparam T Underlying type of elements stored in the vector
/// @tparam Dims Number of dimensions (elements)
template <Arithmetic T, SizeType Dims>
struct Vector : internal::VectorStorage<internal::NoCv<T>, Dims> {
  /// @brief Underlying type of elements stored in the vector
  using Scalar = internal::NoCvRef<T>;
  /// @brief Type used to perform pointer arithmetic
  using SizeType = ::math::SizeType;

  /// @brief Dimensions (elements) in the vector
  constexpr static auto kDims = Dims;

  using internal::VectorStorage<Scalar, Dims>::VectorStorage;
  using Storage = internal::VectorStorage<Scalar, Dims>;
  using Storage::data_;

  using Tag = internal::VectorTag;

  /// @brief Fill Vector with single scalar
  /// 
  /// @param scalar Value to fill Vector with
  constexpr explicit Vector(Scalar const scalar) noexcept { data_.fill(scalar); }

  template <typename U>
  [[nodiscard]] constexpr static declauto From(U *data) noexcept { return std::launder(reinterpret_cast<::math::Vector<T, Dims> *>(data)); }

  /// @brief Syntactic sugar dereferencing this pointer to get a reference
  [[nodiscard]] constexpr declauto self()       noexcept { return *this; }
  /// @brief Syntactic sugar dereferencing this pointer to get a const reference
  [[nodiscard]] constexpr declauto self() const noexcept { return *this; }

  /// @brief Access element of vector
  /// 
  /// @param pos Index of element to access
  /// @return Reference to the element
  [[nodiscard]] constexpr declauto operator[](SizeType const pos)       noexcept { return data_[pos]; }
  /// @brief Access element of vector
  /// 
  /// @param pos Index of element to access
  /// @return Copy of the element
  [[nodiscard]] constexpr Scalar   operator[](SizeType const pos) const noexcept { return data_[pos]; }

  /// @brief Access element of vector
  /// @note The at function performs bounds-checking and may throw, use operator[] for unchecked access instead.
  /// 
  /// @param pos Index of element to access
  /// @return Reference to the element
  /// @throw std::out_of_range if pos is out of bounds
  [[nodiscard]] constexpr declauto at(SizeType const pos) {
    if (pos >= kDims) throw std::out_of_range("invalid vector subscript");
    return data_[pos];
  }

  /// @brief Access element of vector
  /// @note The at function performs bounds-checking and may throw, use operator[] for unchecked access instead.
  /// 
  /// @param pos Index of element to access
  /// @return Copy of the element
  /// @throw std::out_of_range if pos is out of bounds
  [[nodiscard]] constexpr Scalar at(SizeType const pos) const {
    if (pos >= kDims) throw std::out_of_range("invalid vector subscript");
    return data_[pos];
  }

  /// @brief Negate elements of vector, compue its opposite vector
  /// 
  /// @return Negated copy of vector
  [[nodiscard]] constexpr auto operator-()       noexcept { return self() * static_cast<Scalar>(-1); }
  /// @brief Negate elements of vector, compue its opposite vector
  /// 
  /// @return Negated copy of vector
  [[nodiscard]] constexpr auto operator-() const noexcept { return self() * static_cast<Scalar>(-1); }

  /// @brief Pointer to the vector's first element in memory
  [[nodiscard]] constexpr declauto data()       noexcept { return std::data(data_); }
  /// @brief Const pointer to the vector's first element in memory
  [[nodiscard]] constexpr declauto data() const noexcept { return std::data(data_); }

  /// @brief Access std::array storage for structured-bindings
  [[nodiscard]] constexpr auto       &raw()       noexcept { return data_; }
  /// @brief Access std::array storage for structured-bindings
  [[nodiscard]] constexpr auto const &raw() const noexcept { return data_; }

  /// @brief Get size of vector
  [[nodiscard]] constexpr auto size() const noexcept { return kDims; }
  /// @brief Get dimensions of vector
  [[nodiscard]] constexpr auto dims() const noexcept { return kDims; }

  [[nodiscard]] constexpr declauto cbegin() const noexcept { return std::cbegin(data_); }
  [[nodiscard]] constexpr declauto cend()   const noexcept { return std::cend(data_); }

  [[nodiscard]] constexpr declauto begin()       noexcept { return std::begin(data_); }
  [[nodiscard]] constexpr declauto begin() const noexcept { return cbegin(); }

  [[nodiscard]] constexpr declauto end()       noexcept { return std::end(data_); }
  [[nodiscard]] constexpr declauto end() const noexcept { return cend(); }

  // @brief Compute |v|², synonymous with dot(v,v)
  [[nodiscard]] constexpr Scalar length_squared()    const noexcept { return dot(); }
  // @brief Compute |v|
  [[nodiscard]] constexpr Scalar length()            const noexcept { return ::math::ct_sqrt(length_squared()); }
  // @brief Compute 1/|v|
  [[nodiscard]] constexpr Scalar reciprocal_length() const noexcept { return static_cast<Scalar>(1) / length(); }
  // @brief Compute distance to other vector
  [[nodiscard]] constexpr Scalar distance(Vector const &other) const noexcept { return (self() - other).length(); }

  // @brief Resize vector
                constexpr void resize (Scalar const scale)       noexcept {        self() *= (reciprocal_length() * scale); }
  // @brief Resize vector into a copy
  [[nodiscard]] constexpr auto resized(Scalar const scale) const noexcept { return self() *  (reciprocal_length() * scale); }

  // @brief Normalize vector
                constexpr void normalize ()       noexcept {        resize (1); }

  // @brief Normalize vector into a copy
  [[nodiscard]] constexpr auto normalized() const noexcept { return resized(1); }
  
  /// @brief Computes the dot product of the vector with itself
  [[nodiscard]] constexpr Scalar dot()                    const noexcept { return dot(self()); }
  /// @brief Computes the dot product of this and another vector
  /// 
  /// @param other The other vector
  /// @return dot product of this and another vector
  [[nodiscard]] constexpr Scalar dot(Vector const &other) const noexcept { return ::math::dot(self(), other); }

  // @brief Checks if all elements of the vector are equal zero
  // @note For floating point scalars |v|² < epsilon is used to determine zero-equality
  [[nodiscard]] constexpr bool is_zero() const noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
      return dot() < ::math::kEpsilon<Scalar>;
    } else /* std::is_integral_v<T> */ {
      return self() == std::declval<Vector>();
    }
  }

  [[nodiscard]] constexpr declauto    x() noexcept requires(kDims >= 1) { return self()[0]; }
  [[nodiscard]] constexpr declauto    y() noexcept requires(kDims >= 2) { return self()[1]; }
  [[nodiscard]] constexpr declauto    z() noexcept requires(kDims >= 3) { return self()[2]; }
  [[nodiscard]] constexpr declauto    w() noexcept requires(kDims >= 4) { return self()[3]; }
  [[nodiscard]] constexpr declauto   xy() noexcept requires(kDims >= 2) { return *Vector<Scalar, 2>::From(data() + 0); }
  [[nodiscard]] constexpr declauto   yz() noexcept requires(kDims >= 3) { return *Vector<Scalar, 2>::From(data() + 1); }
  [[nodiscard]] constexpr declauto   zw() noexcept requires(kDims >= 4) { return *Vector<Scalar, 2>::From(data() + 2); }
  [[nodiscard]] constexpr declauto  xyz() noexcept requires(kDims >= 3) { return *Vector<Scalar, 3>::From(data() + 0); }
  [[nodiscard]] constexpr declauto  yzw() noexcept requires(kDims >= 4) { return *Vector<Scalar, 3>::From(data() + 1); }
  [[nodiscard]] constexpr declauto xyzw() noexcept requires(kDims >= 4) { return *Vector<Scalar, 4>::From(data() + 0); }

  [[nodiscard]] constexpr Vector cross(Vector<T, 3> const &other) const noexcept requires(kDims == 3) { return math::cross(self(), other); }
};

/// @brief Ensures all types in parameter pack are the same as the first one
template <typename T, typename... Ts>
struct StrictParameterTypes {
  static_assert(std::conjunction_v<std::is_same<internal::NoCvRef<T>, Ts>...>, "All values are required to be of the same type.");

  using U = internal::NoCvRef<T>;
};

/// @brief CTAD deduction guidelines for Vector
template <Arithmetic T, Arithmetic... Ts>
Vector(T, Ts...) -> Vector<typename StrictParameterTypes<T, Ts...>::U, 1 + sizeof...(Ts)>;

namespace internal {
template <Arithmetic T, SizeType Rows, SizeType Columns>
struct Matrix {
  static_assert(Rows && Columns, "Rows and Columns are required to be non-zero.");

  /// @brief Underlying type of elements stored in the matrix
  using Scalar = internal::NoCvRef<T>;
  /// @brief Type used to perform pointer arithmetic
  using SizeType = ::math::SizeType;

  /// @brief Rows in the matrix
  constexpr static auto kRows = Rows;
  /// @brief Columns in the matrix
  constexpr static auto kColumns = Columns;
  /// @brief Elements in the matrix
  constexpr static auto kElements = kRows * kColumns;

  /// @brief Construct default initialized matrix
  constexpr Matrix() noexcept = default;

  /// @brief Fill all values in the matrix with scalar s
  ///
  /// @param s scalar to fill matrix with
  explicit constexpr Matrix(Scalar const s) noexcept {
    for (auto &&x : data_)
      x = Vector<T, Rows>{s};
  }

  std::array<Vector<T, Rows>, Columns> data_{};
};
}

namespace aliases {
using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

using Vec2u32 = Vector<std::uint32_t, 2>;
using Vec3u32 = Vector<std::uint32_t, 3>;
using Vec4u32 = Vector<std::uint32_t, 4>;

using Vec2u64 = Vector<std::uint64_t, 2>;
using Vec3u64 = Vector<std::uint64_t, 3>;
using Vec4u64 = Vector<std::uint64_t, 4>;

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

using Vec2u = Vec2u32;
using Vec3u = Vec3u32;
using Vec4u = Vec4u32;

template <typename T, SizeType Dims>
using Vec = Vector<T, Dims>;
template <typename T>
using Vec2 = Vec<T, 2>;
template <typename T>
using Vec3 = Vec<T, 3>;
template <typename T>
using Vec4 = Vec<T, 4>;
}  // namespace aliases

#ifndef KMATH_NO_ALIASES
using namespace aliases;
#endif  // KMATH_NO_ALIASES

/// @brief Converts radians to degrees
template <std::floating_point T>
[[nodiscard]] constexpr inline T const rad_to_deg(T const rad) noexcept {
  return static_cast<T>(rad * 180.0L / kPi<long double>);
}

/// @brief Converts degrees to radians
template <std::floating_point T>
[[nodiscard]] constexpr inline T const deg_to_rad(T const deg) noexcept {
  return static_cast<T>(deg * kPi<long double> / 180.0L);
}

namespace tests {
template <Arithmetic Scalar, SizeType Dims>
struct CheckAlignment {
  static_assert(
      alignof(Vector<Scalar, Dims>) == ::math::internal::align(sizeof(Scalar) * Dims),
      "Vector<Scalar, Dims> is incorrectly aligned.");
};

#define KMATH_CHECK_ALIGNMENT_ONE(Type, Name, Dims)                        \
  [[maybe_unused]] inline KMATH_CXX20_CONSTINIT CheckAlignment<Type, Dims> \
      check_algn_##Name##_##Dims{};

#define KMATH_CHECK_ALIGNMENT(Type, Name)  \
  KMATH_CHECK_ALIGNMENT_ONE(Type, Name, 0) \
  KMATH_CHECK_ALIGNMENT_ONE(Type, Name, 1) \
  KMATH_CHECK_ALIGNMENT_ONE(Type, Name, 2) \
  KMATH_CHECK_ALIGNMENT_ONE(Type, Name, 3) \
  KMATH_CHECK_ALIGNMENT_ONE(Type, Name, 4)

KMATH_CHECK_ALIGNMENT(unsigned         int, uint)
KMATH_CHECK_ALIGNMENT(unsigned long   long,  ull)
KMATH_CHECK_ALIGNMENT(               float,  flt)
KMATH_CHECK_ALIGNMENT(              double,  dbl)
KMATH_CHECK_ALIGNMENT(         long double, ldbl)

#undef KMATH_CHECK_ALIGNMENT
#undef KMATH_CHECK_ALIGNMENT_ONE
}  // namespace tests
}  // namespace math

#if __has_include(<iostream>) && defined(KMATH_IOSTREAM)
std::ostream &operator<<(std::ostream &os, ::math::AVector auto &&vec) {
  // TODO: Create string conversion functions for math::Vector, math::Matrix, math::Quaternion
  os << '(';
  for (::math::SizeType i{}; auto &&scalar : vec) {
    os << scalar << ((++i != vec.size()) ? ", " : ")");
  }
  return os;
}
#endif  // __has_include(iostream)

#undef KMATH_CXX20_CONSTINIT
#undef KMATH_CXX20_CONSTEVAL
#undef KMATH_CXX20_NO_UNIQUE_ADDR

#endif  // KMATH_MATH_HPP_
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
[[maybe_unused]] KMATH_CXX20_CONSTINIT auto kLicense = R"(The kMath Project
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

#if __cpp_lib_bit_cast >= 201806L
# include <bit>
#else
# include <cstring>
#endif  // __cpp_lib_bit_cast >= 201806L

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

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
# define declauto decltype(auto)  // PR number assigned shortly
#endif                            // declauto

namespace math {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/*========================*/
/* Mathematical constants */
/*========================*/
// Euler's number constant
template <Arithmetic T>
inline constexpr T kEulersNumber{2.71828182845904523536L};
// log2(e) constant
template <Arithmetic T>
inline constexpr T kLog2e{1.44269504088896340736L};
// log10(e) constant
template <Arithmetic T>
inline constexpr T kLog10e{0.434294481903251827651L};
// ln(2) constant
template <Arithmetic T>
inline constexpr T kLn2{0.693147180559945309417L};
// ln(10) constant
template <Arithmetic T>
inline constexpr T kLn10{2.30258509299404568402L};
// π constant
template <Arithmetic T>
inline constexpr T kPi{3.14159265358979323846L};
// π/2 constant
template <Arithmetic T>
inline constexpr T kPiOver2{1.57079632679489661923L};
// π/4 constant
template <Arithmetic T>
inline constexpr T kPiOver4{0.785398163397448309616L};
// √2 constant
template <Arithmetic T>
inline constexpr T kSqrt2{1.41421356237309504880L};
// 1/π constant
template <Arithmetic T>
inline constexpr T kOneOverPi{0.318309886183790671538L};
// 2/π constant
template <Arithmetic T>
inline constexpr T kTwoOverPi{0.636619772367581343076L};
// 2/√π constant
template <Arithmetic T>
inline constexpr T kTwoOverSqrtPi{1.12837916709551257390L};
// 1/√2 constant
template <Arithmetic T>
inline constexpr T kTwoOverSqrt2{0.707106781186547524401L};
// quiet nan constant
template <Arithmetic T>
inline constexpr T kQuietNan = std::numeric_limits<T>::quiet_NaN();
// signal nan constant
template <Arithmetic T>
inline constexpr T kSignalingNan = std::numeric_limits<T>::signaling_NaN();
// nan constant
template <Arithmetic T>
inline constexpr T kNan = kQuietNan<T>;
// inf constant
template <Arithmetic T>
inline constexpr T kInf = std::numeric_limits<T>::infinity();
// inf constant
template <Arithmetic T>
inline constexpr T kEpsilon = std::numeric_limits<T>::epsilon();

using SizeType = std::array<std::nullptr_t, 1>::size_type;

template <Arithmetic T, SizeType Dims>
struct VectorStorage;

template <Arithmetic T, SizeType Dims>
struct Vector;

// clang-format off
namespace internal {
struct NonCopyAssignable {
  auto &operator=(NonCopyAssignable const &) = delete;
  auto &operator=(NonCopyAssignable) = delete;
};

struct NonMoveAssignable {
  auto &operator=(NonMoveAssignable &&) = delete;
};

struct NonCopyConstructible {
  NonCopyConstructible(NonCopyConstructible const &) = delete;
};

struct NonMoveConstructible {
  NonMoveConstructible(NonMoveConstructible &&) = delete;
};

struct NonDefaultConstructible {
  NonDefaultConstructible() = delete;
};

struct Tag : NonDefaultConstructible, NonCopyConstructible, NonMoveConstructible // Non-constructible
                                    , NonCopyAssignable   , NonMoveAssignable    // Non-assignable
{};

static_assert(!std::is_constructible_v<Tag>      && !std::is_assignable_v<Tag, Tag>
           && !std::is_copy_constructible_v<Tag> && !std::is_copy_assignable_v<Tag>
           && !std::is_move_constructible_v<Tag> && !std::is_move_assignable_v<Tag>);

struct VectorTag     : Tag{};
struct MatrixTag     : Tag{};
struct QuaternionTag : Tag{};

template <typename T> using NoCvRef = std::remove_cvref_t<T>;
template <typename T> using NoCv    = std::remove_cv_t<T>;
}  // namespace internal

template <typename T> concept AVector     = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::VectorTag>;
template <typename T> concept AMatrix     = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::MatrixTag>;
template <typename T> concept AQuaternion = std::is_same_v<typename internal::NoCvRef<T>::Tag, internal::QuaternionTag>;


namespace internal {
constexpr auto newton_raphson(Arithmetic auto scalar, Arithmetic auto cur, Arithmetic auto prev) noexcept {
  using Scalar = internal::NoCvRef<decltype(scalar)>;
  if (cur == prev)
    return cur;
  else
    return newton_raphson(scalar, static_cast<Scalar>(.5) * (cur + scalar / cur), cur);
}

constexpr auto ct_sqrt(Arithmetic auto scalar) noexcept {
  using Scalar = internal::NoCvRef<decltype(scalar)>;
  return scalar >= kEpsilon<Scalar> && scalar < kInf<Scalar>
    ? newton_raphson(scalar, scalar, Scalar{})
    : kQuietNan<Scalar>;
}

constexpr std::size_t align(const std::size_t size) noexcept {
    if (size == 0) return 1;
    if (size <= sizeof(std::uint32_t)) return alignof(std::uint32_t);
    if (size <= sizeof(std::uint64_t)) return alignof(std::uint64_t);
    return alignof(std::max_align_t);
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
// clang-format on
}  // namespace internal

constexpr auto ct_sqrt(Arithmetic auto const scalar) noexcept {
  if (std::is_constant_evaluated()) {
    return ::math::internal::ct_sqrt(scalar);
  } else {
    return std::sqrt(scalar);
  }
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
constexpr declauto implement_arithmetic(auto &&lhs, auto &&rhs,
                                        auto &&op) noexcept {
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

#define KMATH_IMPL_ARITHMETIC_IMPL(ShouldCopy, Token, Dims, Op)                                                \
  template <typename U, typename V>                                                                            \
    requires ::math::AVector<U> || ::math::AVector<V>                                                          \
  constexpr declauto operator Token(U &&u, V &&v) noexcept {                                                   \
    return ::math::internal::implement_arithmetic<ShouldCopy>(std::forward<U>(u), std::forward<V>(v), Op<>{}); \
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

template <Arithmetic T, SizeType Dims>
struct Vector : internal::VectorStorage<internal::NoCv<T>, Dims> {
  constexpr static auto kDims = Dims;
  using Scalar = internal::NoCvRef<T>;
  using Derived  = ::math::Vector<Scalar, Dims>;
  using SizeType = ::math::SizeType;

  using internal::VectorStorage<Scalar, Dims>::VectorStorage;
  using Storage = internal::VectorStorage<Scalar, Dims>;
  using Storage::data_;

  using Tag = internal::VectorTag;

  constexpr explicit Vector(Scalar const scalar) noexcept { data_.fill(scalar); }

  template <typename U>
  [[deprecated("This function technically invokes undefined behavior (UB), only use this if there is no available alternative!")]]
  [[nodiscard]] constexpr static decltype(auto) From(U *data) noexcept {
#if __cpp_lib_bit_cast >= 201806L
    return *std::bit_cast<Vector<T, Dims> *>(data);
#else
    Vector<T, Dims> result;
    std::memcpy(&result, data, sizeof(result));
    return result;
#endif  // __cpp_lib_bit_cast >= 201806L
  }

  [[nodiscard]] constexpr declauto self()       noexcept { return *this; }
  [[nodiscard]] constexpr declauto self() const noexcept { return *this; }

  [[nodiscard]] constexpr declauto operator[](SizeType const pos)       noexcept { return data_[pos]; }
  [[nodiscard]] constexpr Scalar   operator[](SizeType const pos) const noexcept { return data_[pos]; }

  [[nodiscard]] explicit constexpr operator Scalar *()             noexcept { return std::data(data_); }
  [[nodiscard]] explicit constexpr operator Scalar *() const noexcept { return std::data(data_); }

  [[nodiscard]] constexpr auto operator-()       noexcept { return self() * static_cast<Scalar>(-1); }
  [[nodiscard]] constexpr auto operator-() const noexcept { return self() * static_cast<Scalar>(-1); }

  [[nodiscard]] constexpr declauto data()       noexcept { return std::data(data_); }
  [[nodiscard]] constexpr declauto data() const noexcept { return std::data(data_); }

  [[nodiscard]] constexpr auto size() const noexcept { return kDims; }
  [[nodiscard]] constexpr auto dims() const noexcept { return kDims; }

  [[nodiscard]] constexpr declauto cbegin() const noexcept { return std::cbegin(data_); }
  [[nodiscard]] constexpr declauto cend()   const noexcept { return std::cend(data_); }

  [[nodiscard]] constexpr declauto begin()       noexcept { return std::begin(data_); }
  [[nodiscard]] constexpr declauto begin() const noexcept { return cbegin(); }

  [[nodiscard]] constexpr declauto end()       noexcept { return std::end(data_); }
  [[nodiscard]] constexpr declauto end() const noexcept { return cend(); }

  template <Arithmetic U>
  [[nodiscard]] constexpr auto truncate() const noexcept { return truncate<U>(std::make_index_sequence<Dims>{}); }

  [[nodiscard]] constexpr Scalar length_squared() const noexcept { return dot(); }
  [[nodiscard]] constexpr Scalar length() const noexcept { return ::math::ct_sqrt(length_squared()); }
  [[nodiscard]] constexpr Scalar reciprocal_length() const noexcept { return static_cast<Scalar>(1) / length(); }

  [[nodiscard]] constexpr Scalar distance(Derived const &other) const noexcept { return (self() - other).length(); }

  [[nodiscard]] constexpr auto resized(Scalar const scale) const { return self() * (reciprocal_length() * scale); }

                constexpr void normalize()  const noexcept { self() *= reciprocal_length(); }
  [[nodiscard]] constexpr auto normalized() const noexcept { return resized(1); }
  
  [[nodiscard]] constexpr Scalar dot(Vector const &other) const noexcept { return ::math::dot(self(), other); }
  [[nodiscard]] constexpr Scalar dot() const noexcept { return dot(self()); }

  [[nodiscard]] constexpr bool is_zero() const noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
      return dot() < ::math::kEpsilon<Scalar>;
    } else /* std::is_integral_v<T> */ {
      return self() == std::declval<Derived>();
    }
  }

 private:
  // TODO: Finish implementing this
  [[nodiscard]] constexpr declauto x()    noexcept requires(kDims >= 1) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto y()    noexcept requires(kDims >= 2) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto z()    noexcept requires(kDims >= 3) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto w()    noexcept requires(kDims >= 4) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto xy()   noexcept requires(kDims >= 2) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto yz()   noexcept requires(kDims >= 3) { return Vector{&Storage::data_[1]}; }
  [[nodiscard]] constexpr declauto zw()   noexcept requires(kDims >= 4) { return Vector{&Storage::data_[2]}; }
  [[nodiscard]] constexpr declauto xyz()  noexcept requires(kDims >= 3) { return Vector{&Storage::data_[0]}; }
  [[nodiscard]] constexpr declauto yzw()  noexcept requires(kDims >= 4) { return Vector{&Storage::data_[1]}; }
  [[nodiscard]] constexpr declauto xyzw() noexcept requires(kDims >= 4) { return Vector{&Storage::data_[0]}; }

  [[nodiscard]] constexpr Vector cross(Vector<T, 3> const &other) const noexcept requires(kDims == 3) { return math::cross(self(), other); }

  template <Arithmetic U, std::size_t... Is>
  [[nodiscard]] constexpr auto truncate(std::index_sequence<Is...>) const noexcept {
    return U{static_cast<typename U::Scalar>(data_[Is])...};
  }
};

// Ensure all types in parameter pack are the same as the first one
template <typename T, typename... Ts>
struct StrictParameterTypes {
  static_assert(std::conjunction_v<std::is_same<internal::NoCvRef<T>, Ts>...>, "All values are required to be of the same type.");

  using U = internal::NoCvRef<T>;
};

// CTAD deduction guidelines for Vector
template <Arithmetic T, Arithmetic... Ts>
Vector(T, Ts...) -> Vector<typename StrictParameterTypes<T, Ts...>::U, 1 + sizeof...(Ts)>;

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

template <std::floating_point T>
constexpr inline T const rad_to_deg(T const rad) {
  return rad * static_cast<T>(180) / kPi<T>;
}

template <std::floating_point T>
constexpr inline T const deg_to_rad(T const deg) {
  return deg * kPi<T> / static_cast<T>(180);
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

KMATH_CHECK_ALIGNMENT(unsigned int, uint)
KMATH_CHECK_ALIGNMENT(unsigned long long, ull)
KMATH_CHECK_ALIGNMENT(float, flt)
KMATH_CHECK_ALIGNMENT(double, dbl)
KMATH_CHECK_ALIGNMENT(long double, ldbl)

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
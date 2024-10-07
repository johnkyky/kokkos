//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_HOST_EXP_ITERATE_HPP
#define KOKKOS_HOST_EXP_ITERATE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP
#endif

#ifdef KOKKOS_MDRANGE_IVDEP
#define KOKKOS_ENABLE_IVDEP_MDRANGE _Pragma("ivdep")
#else
#define KOKKOS_ENABLE_IVDEP_MDRANGE
#endif

#include <algorithm>
#include <iostream>

namespace Kokkos {
namespace Impl {

#define KOKKOS_IMPL_APPLY(func, ...) func(__VA_ARGS__);

/* Right layouy loop implementation */
#define KOKKOS_IMPL_LOOP_1R(func, type, lower, upper, index, ...)          \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                              \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]); \
       ++i0) {                                                             \
    KOKKOS_IMPL_APPLY(func, __VA_ARGS__, i0)                               \
  }

#define KOKKOS_IMPL_LOOP_2R(func, type, lower, upper, index, ...)          \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]); \
       ++i1) {                                                             \
    KOKKOS_IMPL_LOOP_1R(func, type, lower, upper, index + 1,               \
                        i1 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_3R(func, type, lower, upper, index, ...)          \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]); \
       ++i2) {                                                             \
    KOKKOS_IMPL_LOOP_2R(func, type, lower, upper, index + 1,               \
                        i2 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

/* Left layouy loop implementation */
#define KOKKOS_IMPL_LOOP_1L(func, type, lower, upper, index, ...)          \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                              \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]); \
       ++i0) {                                                             \
    KOKKOS_IMPL_APPLY(func, i0, __VA_ARGS__)                               \
  }

#define KOKKOS_IMPL_LOOP_2L(func, type, lower, upper, index, ...)          \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]); \
       ++i1) {                                                             \
    KOKKOS_IMPL_LOOP_1L(func, type, lower, upper, index - 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i1)                     \
  }

#define KOKKOS_IMPL_LOOP_3L(func, type, lower, upper, index, ...)          \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]); \
       ++i2) {                                                             \
    KOKKOS_IMPL_LOOP_2L(func, type, lower, upper, index - 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i2)                     \
  }

// ------------------------------------------------------------------ //

// Structs for calling loops
template <int Rank, typename IType, bool IsLeft, typename Tagged,
          typename Enable = void>
struct Loop_Type;

// Rank = 1
template <typename IType, bool IsLeft>
struct Loop_Type<1, IType, IsLeft, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i0 = (IType)lower[index]; i0 < static_cast<IType>(upper[index]);
         ++i0) {
      KOKKOS_IMPL_APPLY(func, i0)
    }
  }
};

// Rank = 2
template <typename IType>
struct Loop_Type<2, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_1R(func, IType, lower, upper, index + 1, i1)
    }
  }
};

template <typename IType>
struct Loop_Type<2, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 1;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_1L(func, IType, lower, upper, index - 1, i1)
    }
  }
};

// Rank = 3
template <typename IType>
struct Loop_Type<3, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_2R(func, IType, lower, upper, index + 1, i2)
    }
  }
};

template <typename IType>
struct Loop_Type<3, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 2;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_2L(func, IType, lower, upper, index - 1, i2)
    }
  }
};
// end Structs for calling loops

template <typename RP, typename Functor, typename Tag = void,
          typename ValueType = void, typename Enable = void>
struct HostIterate;

// For ParallelFor
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterate<RP, Functor, Tag, ValueType,
                   std::enable_if_t<std::is_void<ValueType>::value>> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  inline HostIterate(RP const& rp) : m_rp(rp) {}

  inline void operator()() const {
    Loop_Type<RP::rank, index_type, (RP::inner_direction == Iterate::Left),
              Tag>::apply(m_rp.m_func, m_rp.m_lower, m_rp.m_upper);
  }

  RP const m_rp;
  std::conditional_t<std::is_void<Tag>::value, int, Tag> m_tag;
};

// ------------------------------------------------------------------ //
#undef PRINT_FIRST_TWO
#undef KOKKOS_IMPL_APPLY
#undef KOKKOS_IMPL_LOOP_1R
#undef KOKKOS_IMPL_LOOP_2R
#undef KOKKOS_IMPL_LOOP_3R
#undef KOKKOS_IMPL_LOOP_4R
#undef KOKKOS_IMPL_LOOP_5R
#undef KOKKOS_IMPL_LOOP_6R
#undef KOKKOS_IMPL_LOOP_7R
#undef KOKKOS_IMPL_LOOP_8R

#undef KOKKOS_IMPL_LOOP_1L
#undef KOKKOS_IMPL_LOOP_2L
#undef KOKKOS_IMPL_LOOP_3L
#undef KOKKOS_IMPL_LOOP_4L
#undef KOKKOS_IMPL_LOOP_5L
#undef KOKKOS_IMPL_LOOP_6L
#undef KOKKOS_IMPL_LOOP_7L
#undef KOKKOS_IMPL_LOOP_8L

}  // namespace Impl
}  // namespace Kokkos

#endif

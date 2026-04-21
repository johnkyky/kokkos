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

/* Non Tagged loop implementation */
#define KOKKOS_IMPL_APPLY(func, ...) func(__VA_ARGS__);
#define KOKKOS_IMPL_APPLY_REDUX(val, func, ...) func(__VA_ARGS__, val);

/* Right layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_LOOP_1R(func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                              \
  for (type i0 = l0; i0 < u0; ++i0) {                      \
    KOKKOS_IMPL_APPLY(func, __VA_ARGS__ __VA_OPT__(, ) i0) \
  }

#define KOKKOS_IMPL_LOOP_2R(func, type, l1, l0, u1, u0, ...)               \
  for (type i1 = l1; i1 < u1; ++i1) {                                      \
    KOKKOS_IMPL_LOOP_1R(func, type, l0, u0, __VA_ARGS__ __VA_OPT__(, ) i1) \
  }

#define KOKKOS_IMPL_LOOP_3R(func, type, l2, l1, l0, u2, u1, u0, ...) \
  for (type i2 = l2; i2 < u2; ++i2) {                                \
    KOKKOS_IMPL_LOOP_2R(func, type, l1, l0, u1, u0,                  \
                        __VA_ARGS__ __VA_OPT__(, ) i2)               \
  }

#define KOKKOS_IMPL_LOOP_4R(func, type, l3, l2, l1, l0, u3, u2, u1, u0, ...) \
  for (type i3 = l3; i3 < u3; ++i3) {                                        \
    KOKKOS_IMPL_LOOP_3R(func, type, l2, l1, l0, u2, u1, u0,                  \
                        __VA_ARGS__ __VA_OPT__(, ) i3)                       \
  }

#define KOKKOS_IMPL_LOOP_5R(func, type, l4, l3, l2, l1, l0, u4, u3, u2, u1, \
                            u0, ...)                                        \
  for (type i4 = l4; i4 < u4; ++i4) {                                       \
    KOKKOS_IMPL_LOOP_4R(func, type, l3, l2, l1, l0, u3, u2, u1, u0,         \
                        __VA_ARGS__ __VA_OPT__(, ) i4)                      \
  }

#define KOKKOS_IMPL_LOOP_6R(func, type, l5, l4, l3, l2, l1, l0, u5, u4, u3, \
                            u2, u1, u0, ...)                                \
  for (type i5 = l5; i5 < u5; ++i5) {                                       \
    KOKKOS_IMPL_LOOP_5R(func, type, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0, \
                        __VA_ARGS__ __VA_OPT__(, ) i5)                      \
  }

#define KOKKOS_IMPL_LOOP_7R(func, type, l6, l5, l4, l3, l2, l1, l0, u6, u5, \
                            u4, u3, u2, u1, u0, ...)                        \
  for (type i6 = l6; i6 < u6; ++i6) {                                       \
    KOKKOS_IMPL_LOOP_6R(func, type, l5, l4, l3, l2, l1, l0, u5, u4, u3, u2, \
                        u1, u0, __VA_ARGS__ __VA_OPT__(, ) i6)              \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_LOOP_REDUX_1R(val, func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                         \
  for (type i0 = l0; i0 < u0; ++i0) {                                 \
    KOKKOS_IMPL_APPLY_REDUX(val, func, __VA_ARGS__ __VA_OPT__(, ) i0) \
  }

#define KOKKOS_IMPL_LOOP_REDUX_2R(val, func, type, l1, l0, u1, u0, ...) \
  for (type i1 = l1; i1 < u1; ++i1) {                                   \
    KOKKOS_IMPL_LOOP_REDUX_1R(val, func, type, l0, u0,                  \
                              __VA_ARGS__ __VA_OPT__(, ) i1)            \
  }

#define KOKKOS_IMPL_LOOP_REDUX_3R(val, func, type, l2, l1, l0, u2, u1, u0, \
                                  ...)                                     \
  for (type i2 = l2; i2 < u2; ++i2) {                                      \
    KOKKOS_IMPL_LOOP_REDUX_2R(val, func, type, l1, l0, u1, u0,             \
                              __VA_ARGS__ __VA_OPT__(, ) i2)               \
  }

#define KOKKOS_IMPL_LOOP_REDUX_4R(val, func, type, l3, l2, l1, l0, u3, u2, u1, \
                                  u0, ...)                                     \
  for (type i3 = l3; i3 < u3; ++i3) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_3R(val, func, type, l2, l1, l0, u2, u1, u0,         \
                              __VA_ARGS__ __VA_OPT__(, ) i3)                   \
  }

#define KOKKOS_IMPL_LOOP_REDUX_5R(val, func, type, l4, l3, l2, l1, l0, u4, u3, \
                                  u2, u1, u0, ...)                             \
  for (type i4 = l4; i4 < u4; ++i4) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_4R(val, func, type, l3, l2, l1, l0, u3, u2, u1, u0, \
                              __VA_ARGS__ __VA_OPT__(, ) i4)                   \
  }

#define KOKKOS_IMPL_LOOP_REDUX_6R(val, func, type, l5, l4, l3, l2, l1, l0, u5, \
                                  u4, u3, u2, u1, u0, ...)                     \
  for (type i5 = l5; i5 < u5; ++i5) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_5R(val, func, type, l4, l3, l2, l1, l0, u4, u3, u2, \
                              u1, u0, __VA_ARGS__ __VA_OPT__(, ) i5)           \
  }

#define KOKKOS_IMPL_LOOP_REDUX_7R(val, func, type, l6, l5, l4, l3, l2, l1, l0, \
                                  u6, u5, u4, u3, u2, u1, u0, ...)             \
  for (type i6 = l6; i6 < u6; ++i6) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_6R(val, func, type, l5, l4, l3, l2, l1, l0, u5, u4, \
                              u3, u2, u1, u0, __VA_ARGS__ __VA_OPT__(, ) i6)   \
  }

/* Left layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_LOOP_1L(func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                              \
  for (type i0 = l0; i0 < u0; ++i0) {                      \
    KOKKOS_IMPL_APPLY(func, i0 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_LOOP_2L(func, type, l1, l0, u1, u0, ...)               \
  for (type i1 = l1; i1 < u1; ++i1) {                                      \
    KOKKOS_IMPL_LOOP_1L(func, type, l0, u0, i1 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_LOOP_3L(func, type, l2, l1, l0, u2, u1, u0, ...) \
  for (type i2 = l2; i2 < u2; ++i2) {                                \
    KOKKOS_IMPL_LOOP_2L(func, type, l1, l0, u1, u0,                  \
                        i2 __VA_OPT__(, ) __VA_ARGS__)               \
  }

#define KOKKOS_IMPL_LOOP_4L(func, type, l3, l2, l1, l0, u3, u2, u1, u0, ...) \
  for (type i3 = l3; i3 < u3; ++i3) {                                        \
    KOKKOS_IMPL_LOOP_3L(func, type, l2, l1, l0, u2, u1, u0,                  \
                        i3 __VA_OPT__(, ) __VA_ARGS__)                       \
  }

#define KOKKOS_IMPL_LOOP_5L(func, type, l4, l3, l2, l1, l0, u4, u3, u2, u1, \
                            u0, ...)                                        \
  for (type i4 = l4; i4 < u4; ++i4) {                                       \
    KOKKOS_IMPL_LOOP_4L(func, type, l3, l2, l1, l0, u3, u2, u1, u0,         \
                        i4 __VA_OPT__(, ) __VA_ARGS__)                      \
  }

#define KOKKOS_IMPL_LOOP_6L(func, type, l5, l4, l3, l2, l1, l0, u5, u4, u3, \
                            u2, u1, u0, ...)                                \
  for (type i5 = l5; i5 < u5; ++i5) {                                       \
    KOKKOS_IMPL_LOOP_5L(func, type, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0, \
                        i5 __VA_OPT__(, ) __VA_ARGS__)                      \
  }

#define KOKKOS_IMPL_LOOP_7L(func, type, l6, l5, l4, l3, l2, l1, l0, u6, u5, \
                            u4, u3, u2, u1, u0, ...)                        \
  for (type i6 = l6; i6 < u6; ++i6) {                                       \
    KOKKOS_IMPL_LOOP_6L(func, type, l5, l4, l3, l2, l1, l0, u5, u4, u3, u2, \
                        u1, u0, i6 __VA_OPT__(, ) __VA_ARGS__)              \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_LOOP_REDUX_1L(val, func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                         \
  for (type i0 = l0; i0 < u0; ++i0) {                                 \
    KOKKOS_IMPL_APPLY_REDUX(val, func, i0 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_LOOP_REDUX_2L(val, func, type, l1, l0, u1, u0, ...) \
  for (type i1 = l1; i1 < u1; ++i1) {                                   \
    KOKKOS_IMPL_LOOP_REDUX_1L(val, func, type, l0, u0,                  \
                              i1 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_LOOP_REDUX_3L(val, func, type, l2, l1, l0, u2, u1, u0, \
                                  ...)                                     \
  for (type i2 = l2; i2 < u2; ++i2) {                                      \
    KOKKOS_IMPL_LOOP_REDUX_2L(val, func, type, l1, l0, u1, u0,             \
                              i2 __VA_OPT__(, ) __VA_ARGS__)               \
  }

#define KOKKOS_IMPL_LOOP_REDUX_4L(val, func, type, l3, l2, l1, l0, u3, u2, u1, \
                                  u0, ...)                                     \
  for (type i3 = l3; i3 < u3; ++i3) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_3L(val, func, type, l2, l1, l0, u2, u1, u0,         \
                              i3 __VA_OPT__(, ) __VA_ARGS__)                   \
  }

#define KOKKOS_IMPL_LOOP_REDUX_5L(val, func, type, l4, l3, l2, l1, l0, u4, u3, \
                                  u2, u1, u0, ...)                             \
  for (type i4 = l4; i4 < u4; ++i4) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_4L(val, func, type, l3, l2, l1, l0, u3, u2, u1, u0, \
                              i4 __VA_OPT__(, ) __VA_ARGS__)                   \
  }

#define KOKKOS_IMPL_LOOP_REDUX_6L(val, func, type, l5, l4, l3, l2, l1, l0, u5, \
                                  u4, u3, u2, u1, u0, ...)                     \
  for (type i5 = l5; i5 < u5; ++i5) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_5L(val, func, type, l4, l3, l2, l1, l0, u4, u3, u2, \
                              u1, u0, i5 __VA_OPT__(, ) __VA_ARGS__)           \
  }

#define KOKKOS_IMPL_LOOP_REDUX_7L(val, func, type, l6, l5, l4, l3, l2, l1, l0, \
                                  u6, u5, u4, u3, u2, u1, u0, ...)             \
  for (type i6 = l6; i6 < u6; ++i6) {                                          \
    KOKKOS_IMPL_LOOP_REDUX_6L(val, func, type, l5, l4, l3, l2, l1, l0, u5, u4, \
                              u3, u2, u1, u0, i6 __VA_OPT__(, ) __VA_ARGS__)   \
  }

/* Tagged loop implementation */
#define KOKKOS_IMPL_TAGGED_APPLY(tag, func, ...) func(tag, __VA_ARGS__);
#define KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func, ...) \
  func(tag, __VA_ARGS__, val);

/* Right layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_TAGGED_LOOP_1R(tag, func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                          \
  for (type i0 = l0; i0 < u0; ++i0) {                                  \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, __VA_ARGS__ __VA_OPT__(, ) i0) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_2R(tag, func, type, l1, l0, u1, u0, ...) \
  for (type i1 = l1; i1 < u1; ++i1) {                                    \
    KOKKOS_IMPL_TAGGED_LOOP_1R(tag, func, type, l0, u0,                  \
                               __VA_ARGS__ __VA_OPT__(, ) i1)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_3R(tag, func, type, l2, l1, l0, u2, u1, u0, \
                                   ...)                                     \
  for (type i2 = l2; i2 < u2; ++i2) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_2R(tag, func, type, l1, l0, u1, u0,             \
                               __VA_ARGS__ __VA_OPT__(, ) i2)               \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_4R(tag, func, type, l3, l2, l1, l0, u3, u2, \
                                   u1, u0, ...)                             \
  for (type i3 = l3; i3 < u3; ++i3) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_3R(tag, func, type, l2, l1, l0, u2, u1, u0,     \
                               __VA_ARGS__ __VA_OPT__(, ) i3)               \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_5R(tag, func, type, l4, l3, l2, l1, l0, u4, \
                                   u3, u2, u1, u0, ...)                     \
  for (type i4 = l4; i4 < u4; ++i4) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_4R(tag, func, type, l3, l2, l1, l0, u3, u2, u1, \
                               u0, __VA_ARGS__ __VA_OPT__(, ) i4)           \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_6R(tag, func, type, l5, l4, l3, l2, l1, l0, \
                                   u5, u4, u3, u2, u1, u0, ...)             \
  for (type i5 = l5; i5 < u5; ++i5) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_5R(tag, func, type, l4, l3, l2, l1, l0, u4, u3, \
                               u2, u1, u0, __VA_ARGS__ __VA_OPT__(, ) i5)   \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_7R(tag, func, type, l6, l5, l4, l3, l2, l1, \
                                   l0, u6, u5, u4, u3, u2, u1, u0, ...)     \
  for (type i6 = l6; i6 < u6; ++i6) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_6R(tag, func, type, l5, l4, l3, l2, l1, l0, u5, \
                               u4, u3, u2, u1, u0,                          \
                               __VA_ARGS__ __VA_OPT__(, ) i6)               \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(tag, val, func, type, l0, u0, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                               \
  for (type i0 = l0; i0 < u0; ++i0) {                                       \
    KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func,                          \
                                   __VA_ARGS__ __VA_OPT__(, ) i0)           \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(tag, val, func, type, l1, l0, u1, u0, \
                                         ...)                                  \
  for (type i1 = l1; i1 < u1; ++i1) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(tag, val, func, type, l0, u0,             \
                                     __VA_ARGS__ __VA_OPT__(, ) i1)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(tag, val, func, type, l2, l1, l0, u2, \
                                         u1, u0, ...)                          \
  for (type i2 = l2; i2 < u2; ++i2) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(tag, val, func, type, l1, l0, u1, u0,     \
                                     __VA_ARGS__ __VA_OPT__(, ) i2)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(tag, val, func, type, l3, l2, l1, l0, \
                                         u3, u2, u1, u0, ...)                  \
  for (type i3 = l3; i3 < u3; ++i3) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(tag, val, func, type, l2, l1, l0, u2, u1, \
                                     u0, __VA_ARGS__ __VA_OPT__(, ) i3)        \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(tag, val, func, type, l4, l3, l2, l1, \
                                         l0, u4, u3, u2, u1, u0, ...)          \
  for (type i4 = l4; i4 < u4; ++i4) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(tag, val, func, type, l3, l2, l1, l0, u3, \
                                     u2, u1, u0,                               \
                                     __VA_ARGS__ __VA_OPT__(, ) i4)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R(tag, val, func, type, l5, l4, l3, l2, \
                                         l1, l0, u5, u4, u3, u2, u1, u0, ...)  \
  for (type i5 = l5; i5 < u5; ++i5) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(tag, val, func, type, l4, l3, l2, l1, l0, \
                                     u4, u3, u2, u1, u0,                       \
                                     __VA_ARGS__ __VA_OPT__(, ) i5)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_7R(tag, val, func, type, l6, l5, l4, l3, \
                                         l2, l1, l0, u6, u5, u4, u3, u2, u1,   \
                                         u0, ...)                              \
  for (type i6 = l6; i6 < u6; ++i6) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R(tag, val, func, type, l5, l4, l3, l2, l1, \
                                     l0, u5, u4, u3, u2, u1, u0,               \
                                     __VA_ARGS__ __VA_OPT__(, ) i6)            \
  }

/* Left layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_TAGGED_LOOP_1L(tag, func, type, l0, u0, ...)       \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                          \
  for (type i0 = l0; i0 < u0; ++i0) {                                  \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, i0 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_2L(tag, func, type, l1, l0, u1, u0, ...) \
  for (type i1 = l1; i1 < u1; ++i1) {                                    \
    KOKKOS_IMPL_TAGGED_LOOP_1L(tag, func, type, l0, u0,                  \
                               i1 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_3L(tag, func, type, l2, l1, l0, u2, u1, u0, \
                                   ...)                                     \
  for (type i2 = l2; i2 < u2; ++i2) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_2L(tag, func, type, l1, l0, u1, u0,             \
                               i2 __VA_OPT__(, ) __VA_ARGS__)               \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_4L(tag, func, type, l3, l2, l1, l0, u3, u2, \
                                   u1, u0, ...)                             \
  for (type i3 = l3; i3 < u3; ++i3) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_3L(tag, func, type, l2, l1, l0, u2, u1, u0,     \
                               i3 __VA_OPT__(, ) __VA_ARGS__)               \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_5L(tag, func, type, l4, l3, l2, l1, l0, u4, \
                                   u3, u2, u1, u0, ...)                     \
  for (type i4 = l4; i4 < u4; ++i4) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_4L(tag, func, type, l3, l2, l1, l0, u3, u2, u1, \
                               u0, i4 __VA_OPT__(, ) __VA_ARGS__)           \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_6L(tag, func, type, l5, l4, l3, l2, l1, l0, \
                                   u5, u4, u3, u2, u1, u0, ...)             \
  for (type i5 = l5; i5 < u5; ++i5) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_5L(tag, func, type, l4, l3, l2, l1, l0, u4, u3, \
                               u2, u1, u0, i5 __VA_OPT__(, ) __VA_ARGS__)   \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_7L(tag, func, type, l6, l5, l4, l3, l2, l1, \
                                   l0, u6, u5, u4, u3, u2, u1, u0, ...)     \
  for (type i6 = l6; i6 < u6; ++i6) {                                       \
    KOKKOS_IMPL_TAGGED_LOOP_6L(tag, func, type, l5, l4, l3, l2, l1, l0, u5, \
                               u4, u3, u2, u1, u0,                          \
                               i6 __VA_OPT__(, ) __VA_ARGS__)               \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(tag, val, func, type, l0, u0, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                               \
  for (type i0 = l0; i0 < u0; ++i0) {                                       \
    KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func,                          \
                                   i0 __VA_OPT__(, ) __VA_ARGS__)           \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(tag, val, func, type, l1, l0, u1, u0, \
                                         ...)                                  \
  for (type i1 = l1; i1 < u1; ++i1) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(tag, val, func, type, l0, u0,             \
                                     i1 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(tag, val, func, type, l2, l1, l0, u2, \
                                         u1, u0, ...)                          \
  for (type i2 = l2; i2 < u2; ++i2) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(tag, val, func, type, l1, l0, u1, u0,     \
                                     i2 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(tag, val, func, type, l3, l2, l1, l0, \
                                         u3, u2, u1, u0, ...)                  \
  for (type i3 = l3; i3 < u3; ++i3) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(tag, val, func, type, l2, l1, l0, u2, u1, \
                                     u0, i3 __VA_OPT__(, ) __VA_ARGS__)        \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(tag, val, func, type, l4, l3, l2, l1, \
                                         l0, u4, u3, u2, u1, u0, ...)          \
  for (type i4 = l4; i4 < u4; ++i4) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(tag, val, func, type, l3, l2, l1, l0, u3, \
                                     u2, u1, u0,                               \
                                     i4 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L(tag, val, func, type, l5, l4, l3, l2, \
                                         l1, l0, u5, u4, u3, u2, u1, u0, ...)  \
  for (type i5 = l5; i5 < u5; ++i5) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(tag, val, func, type, l4, l3, l2, l1, l0, \
                                     u4, u3, u2, u1, u0,                       \
                                     i5 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_7L(tag, val, func, type, l6, l5, l4, l3, \
                                         l2, l1, l0, u6, u5, u4, u3, u2, u1,   \
                                         u0, ...)                              \
  for (type i6 = l6; i6 < u6; ++i6) {                                          \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L(tag, val, func, type, l5, l4, l3, l2, l1, \
                                     l0, u5, u4, u3, u2, u1, u0,               \
                                     i6 __VA_OPT__(, ) __VA_ARGS__)            \
  }

// ------------------------------------------------------------------ //

/* Structs for calling loops */
template <StringAssumption StrAssumption, StringAssumption Backend, int Rank,
          typename IType, bool IsLeft, typename Tagged>
struct Loop_Type;

// Rank = 2 non tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 2, IType, /*LayoutRight*/ false,
                 void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_LOOP_1R(func, IType, l0, u0, i1)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l1, "lower bound 0");
      __builtin_annotation(u1, "upper bound 0");
      __builtin_annotation(l0, "lower bound 1");
      __builtin_annotation(u0, "upper bound 1");

      for (IType i1 = l1; i1 < u1; ++i1) {
        KOKKOS_IMPL_LOOP_1R(func, IType, l0, u0, i1)
      }
    };
    return lambda;
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_LOOP_REDUX_1R(value, func, IType, l0, u0, i1)
    }
  }
};

// Rank = 2 non tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 2, IType, /*LayoutLeft*/ true, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_LOOP_1L(func, IType, l0, u0, i1)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l1, "lower bound 0");
      __builtin_annotation(u1, "upper bound 0");
      __builtin_annotation(l0, "lower bound 1");
      __builtin_annotation(u0, "upper bound 1");

      for (IType i1 = l1; i1 < u1; ++i1) {
        KOKKOS_IMPL_LOOP_1L(func, IType, l0, u0, i1)
      }
    };
    return lambda;
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_LOOP_REDUX_1L(value, func, IType, l0, u0, i1)
    }
  }
};

// Rank = 2 tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 2, IType, /*LayoutRight*/ false,
                 Tagged> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_1R(Tagged(), func, IType, l0, u0, i1)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l1, "lower bound 0");
      __builtin_annotation(u1, "upper bound 0");
      __builtin_annotation(l0, "lower bound 1");
      __builtin_annotation(u0, "upper bound 1");

      for (IType i1 = l1; i1 < u1; ++i1) {
        KOKKOS_IMPL_TAGGED_LOOP_1R(Tagged(), func, IType, l0, u0, i1)
      }
    };
    return lambda;
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[0];
    const IType u1 = static_cast<IType>(upper[0]);
    const IType l0 = (IType)lower[1];
    const IType u0 = static_cast<IType>(upper[1]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(Tagged(), value, func, IType, l0, u0, i1)
    }
  }
};

// Rank = 2 tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 2, IType, /*LayoutLeft*/ true,
                 Tagged> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_1L(Tagged(), func, IType, l0, u0, i1)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l1, "lower bound 0");
      __builtin_annotation(u1, "upper bound 0");
      __builtin_annotation(l0, "lower bound 1");
      __builtin_annotation(u0, "upper bound 1");

      for (IType i1 = l1; i1 < u1; ++i1) {
        KOKKOS_IMPL_TAGGED_LOOP_1L(Tagged(), func, IType, l0, u0, i1)
      }
    };
    return lambda;
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l1, "lower bound 0");
    __builtin_annotation(u1, "upper bound 0");
    __builtin_annotation(l0, "lower bound 1");
    __builtin_annotation(u0, "upper bound 1");

    for (IType i1 = l1; i1 < u1; ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(Tagged(), value, func, IType, l0, u0, i1)
    }
  }
};

// Rank = 3 non tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 3, IType, /*LayoutRight*/ false,
                 void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_LOOP_2R(func, IType, l1, l0, u1, u0, i2)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l2, "lower bound 0");
      __builtin_annotation(u2, "upper bound 0");
      __builtin_annotation(l1, "lower bound 1");
      __builtin_annotation(u1, "upper bound 1");
      __builtin_annotation(l0, "lower bound 2");
      __builtin_annotation(u0, "upper bound 2");

      for (IType i2 = l2; i2 < u2; ++i2) {
        KOKKOS_IMPL_LOOP_2R(func, IType, l1, l0, u1, u0, i2)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_LOOP_REDUX_2R(value, func, IType, l1, l0, u1, u0, i2)
    }
  }
};

// Rank = 3 non tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 3, IType, /*LayoutLeft*/ true, void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_LOOP_2L(func, IType, l1, l0, u1, u0, i2)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l2, "lower bound 0");
      __builtin_annotation(u2, "upper bound 0");
      __builtin_annotation(l1, "lower bound 1");
      __builtin_annotation(u1, "upper bound 1");
      __builtin_annotation(l0, "lower bound 2");
      __builtin_annotation(u0, "upper bound 2");

      for (IType i2 = l2; i2 < u2; ++i2) {
        KOKKOS_IMPL_LOOP_2L(func, IType, l1, l0, u1, u0, i2)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_LOOP_REDUX_2L(value, func, IType, l1, l0, u1, u0, i2)
    }
  }
};

// Rank = 3 tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 3, IType, /*LayoutRight*/ false,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_2R(Tagged(), func, IType, l1, l0, u1, u0, i2)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l2, "lower bound 0");
      __builtin_annotation(u2, "upper bound 0");
      __builtin_annotation(l1, "lower bound 1");
      __builtin_annotation(u1, "upper bound 1");
      __builtin_annotation(l0, "lower bound 2");
      __builtin_annotation(u0, "upper bound 2");

      for (IType i2 = l2; i2 < u2; ++i2) {
        KOKKOS_IMPL_TAGGED_LOOP_2R(Tagged(), func, IType, l1, l0, u1, u0, i2)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[0];
    const IType u2 = static_cast<IType>(upper[0]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[2];
    const IType u0 = static_cast<IType>(upper[2]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(Tagged(), value, func, IType, l1, l0, u1,
                                       u0, i2)
    }
  }
};

// Rank = 3 tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 3, IType, /*LayoutLeft*/ true,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_2L(Tagged(), func, IType, l1, l0, u1, u0, i2)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l2, "lower bound 0");
      __builtin_annotation(u2, "upper bound 0");
      __builtin_annotation(l1, "lower bound 1");
      __builtin_annotation(u1, "upper bound 1");
      __builtin_annotation(l0, "lower bound 2");
      __builtin_annotation(u0, "upper bound 2");

      for (IType i2 = l2; i2 < u2; ++i2) {
        KOKKOS_IMPL_TAGGED_LOOP_2L(Tagged(), func, IType, l1, l0, u1, u0, i2)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l2, "lower bound 0");
    __builtin_annotation(u2, "upper bound 0");
    __builtin_annotation(l1, "lower bound 1");
    __builtin_annotation(u1, "upper bound 1");
    __builtin_annotation(l0, "lower bound 2");
    __builtin_annotation(u0, "upper bound 2");

    for (IType i2 = l2; i2 < u2; ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(Tagged(), value, func, IType, l1, l0, u1,
                                       u0, i2)
    }
  }
};

// Rank = 4 non tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 4, IType, /*LayoutRight*/ false,
                 void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_LOOP_3R(func, IType, l2, l1, l0, u2, u1, u0, i3)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l3, "lower bound 0");
      __builtin_annotation(u3, "upper bound 0");
      __builtin_annotation(l2, "lower bound 1");
      __builtin_annotation(u2, "upper bound 1");
      __builtin_annotation(l1, "lower bound 2");
      __builtin_annotation(u1, "upper bound 2");
      __builtin_annotation(l0, "lower bound 3");
      __builtin_annotation(u0, "upper bound 3");

      for (IType i3 = l3; i3 < u3; ++i3) {
        KOKKOS_IMPL_LOOP_3R(func, IType, l2, l1, l0, u2, u1, u0, i3)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_LOOP_REDUX_3R(value, func, IType, l2, l1, l0, u2, u1, u0, i3)
    }
  }
};

// Rank = 4 non tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 4, IType, /*LayoutLeft*/ true, void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_LOOP_3L(func, IType, l2, l1, l0, u2, u1, u0, i3)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l3, "lower bound 0");
      __builtin_annotation(u3, "upper bound 0");
      __builtin_annotation(l2, "lower bound 1");
      __builtin_annotation(u2, "upper bound 1");
      __builtin_annotation(l1, "lower bound 2");
      __builtin_annotation(u1, "upper bound 2");
      __builtin_annotation(l0, "lower bound 3");
      __builtin_annotation(u0, "upper bound 3");

      for (IType i3 = l3; i3 < u3; ++i3) {
        KOKKOS_IMPL_LOOP_3L(func, IType, l2, l1, l0, u2, u1, u0, i3)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_LOOP_REDUX_3L(value, func, IType, l2, l1, l0, u2, u1, u0, i3)
    }
  }
};

// Rank = 4 tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 4, IType, /*LayoutRight*/ false,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_3R(Tagged(), func, IType, l2, l1, l0, u2, u1, u0,
                                 i3)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l3, "lower bound 0");
      __builtin_annotation(u3, "upper bound 0");
      __builtin_annotation(l2, "lower bound 1");
      __builtin_annotation(u2, "upper bound 1");
      __builtin_annotation(l1, "lower bound 2");
      __builtin_annotation(u1, "upper bound 2");
      __builtin_annotation(l0, "lower bound 3");
      __builtin_annotation(u0, "upper bound 3");

      for (IType i3 = l3; i3 < u3; ++i3) {
        KOKKOS_IMPL_TAGGED_LOOP_3R(Tagged(), func, IType, l2, l1, l0, u2, u1,
                                      u0, i3)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[0];
    const IType u3 = static_cast<IType>(upper[0]);
    const IType l2 = (IType)lower[1];
    const IType u2 = static_cast<IType>(upper[1]);
    const IType l1 = (IType)lower[2];
    const IType u1 = static_cast<IType>(upper[2]);
    const IType l0 = (IType)lower[3];
    const IType u0 = static_cast<IType>(upper[3]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(Tagged(), value, func, IType, l2, l1, l0,
                                       u2, u1, u0, i3)
    }
  }
};

// Rank = 4 tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 4, IType, /*LayoutLeft*/ true,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_3L(Tagged(), func, IType, l2, l1, l0, u2, u1, u0,
                                 i3)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l3, "lower bound 0");
      __builtin_annotation(u3, "upper bound 0");
      __builtin_annotation(l2, "lower bound 1");
      __builtin_annotation(u2, "upper bound 1");
      __builtin_annotation(l1, "lower bound 2");
      __builtin_annotation(u1, "upper bound 2");
      __builtin_annotation(l0, "lower bound 3");
      __builtin_annotation(u0, "upper bound 3");

      for (IType i3 = l3; i3 < u3; ++i3) {
        KOKKOS_IMPL_TAGGED_LOOP_3L(Tagged(), func, IType, l2, l1, l0, u2, u1,
                                      u0, i3)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l3, "lower bound 0");
    __builtin_annotation(u3, "upper bound 0");
    __builtin_annotation(l2, "lower bound 1");
    __builtin_annotation(u2, "upper bound 1");
    __builtin_annotation(l1, "lower bound 2");
    __builtin_annotation(u1, "upper bound 2");
    __builtin_annotation(l0, "lower bound 3");
    __builtin_annotation(u0, "upper bound 3");

    for (IType i3 = l3; i3 < u3; ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(Tagged(), value, func, IType, l2, l1, l0,
                                       u2, u1, u0, i3)
    }
  }
};

// Rank = 5 non tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 5, IType, /*LayoutRight*/ false,
                 void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_LOOP_4R(func, IType, l3, l2, l1, l0, u3, u2, u1, u0, i4)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l4, "lower bound 0");
      __builtin_annotation(u4, "upper bound 0");
      __builtin_annotation(l3, "lower bound 1");
      __builtin_annotation(u3, "upper bound 1");
      __builtin_annotation(l2, "lower bound 2");
      __builtin_annotation(u2, "upper bound 2");
      __builtin_annotation(l1, "lower bound 3");
      __builtin_annotation(u1, "upper bound 3");
      __builtin_annotation(l0, "lower bound 4");
      __builtin_annotation(u0, "upper bound 4");

      for (IType i4 = l4; i4 < u4; ++i4) {
        KOKKOS_IMPL_LOOP_4R(func, IType, l3, l2, l1, l0, u3, u2, u1, u0, i4)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_LOOP_REDUX_4R(value, func, IType, l3, l2, l1, l0, u3, u2, u1,
                                u0, i4)
    }
  }
};

// Rank = 5 non tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 5, IType, /*LayoutLeft*/ true, void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_LOOP_4L(func, IType, l3, l2, l1, l0, u3, u2, u1, u0, i4)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l4, "lower bound 0");
      __builtin_annotation(u4, "upper bound 0");
      __builtin_annotation(l3, "lower bound 1");
      __builtin_annotation(u3, "upper bound 1");
      __builtin_annotation(l2, "lower bound 2");
      __builtin_annotation(u2, "upper bound 2");
      __builtin_annotation(l1, "lower bound 3");
      __builtin_annotation(u1, "upper bound 3");
      __builtin_annotation(l0, "lower bound 4");
      __builtin_annotation(u0, "upper bound 4");

      for (IType i4 = l4; i4 < u4; ++i4) {
        KOKKOS_IMPL_LOOP_4L(func, IType, l3, l2, l1, l0, u3, u2, u1, u0, i4)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_LOOP_REDUX_4L(value, func, IType, l3, l2, l1, l0, u3, u2, u1,
                                u0, i4)
    }
  }
};

// Rank = 5 tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 5, IType, /*LayoutRight*/ false,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_4R(Tagged(), func, IType, l3, l2, l1, l0, u3, u2,
                                 u1, u0, i4)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l4, "lower bound 0");
      __builtin_annotation(u4, "upper bound 0");
      __builtin_annotation(l3, "lower bound 1");
      __builtin_annotation(u3, "upper bound 1");
      __builtin_annotation(l2, "lower bound 2");
      __builtin_annotation(u2, "upper bound 2");
      __builtin_annotation(l1, "lower bound 3");
      __builtin_annotation(u1, "upper bound 3");
      __builtin_annotation(l0, "lower bound 4");
      __builtin_annotation(u0, "upper bound 4");

      for (IType i4 = l4; i4 < u4; ++i4) {
        KOKKOS_IMPL_TAGGED_LOOP_4R(Tagged(), func, IType, l3, l2, l1, l0, u3,
                                      u2, u1, u0, i4)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[0];
    const IType u4 = static_cast<IType>(upper[0]);
    const IType l3 = (IType)lower[1];
    const IType u3 = static_cast<IType>(upper[1]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[3];
    const IType u1 = static_cast<IType>(upper[3]);
    const IType l0 = (IType)lower[4];
    const IType u0 = static_cast<IType>(upper[4]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(Tagged(), value, func, IType, l3, l2, l1,
                                       l0, u3, u2, u1, u0, i4)
    }
  }
};

// Rank = 5 tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 5, IType, /*LayoutLeft*/ true,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_4L(Tagged(), func, IType, l3, l2, l1, l0, u3, u2,
                                 u1, u0, i4)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l4, "lower bound 0");
      __builtin_annotation(u4, "upper bound 0");
      __builtin_annotation(l3, "lower bound 1");
      __builtin_annotation(u3, "upper bound 1");
      __builtin_annotation(l2, "lower bound 2");
      __builtin_annotation(u2, "upper bound 2");
      __builtin_annotation(l1, "lower bound 3");
      __builtin_annotation(u1, "upper bound 3");
      __builtin_annotation(l0, "lower bound 4");
      __builtin_annotation(u0, "upper bound 4");

      for (IType i4 = l4; i4 < u4; ++i4) {
        KOKKOS_IMPL_TAGGED_LOOP_4L(Tagged(), func, IType, l3, l2, l1, l0, u3,
                                      u2, u1, u0, i4)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l4, "lower bound 0");
    __builtin_annotation(u4, "upper bound 0");
    __builtin_annotation(l3, "lower bound 1");
    __builtin_annotation(u3, "upper bound 1");
    __builtin_annotation(l2, "lower bound 2");
    __builtin_annotation(u2, "upper bound 2");
    __builtin_annotation(l1, "lower bound 3");
    __builtin_annotation(u1, "upper bound 3");
    __builtin_annotation(l0, "lower bound 4");
    __builtin_annotation(u0, "upper bound 4");

    for (IType i4 = l4; i4 < u4; ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(Tagged(), value, func, IType, l3, l2, l1,
                                       l0, u3, u2, u1, u0, i4)
    }
  }
};

// Rank = 6 non tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 6, IType, /*LayoutRight*/ false,
                 void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_LOOP_5R(func, IType, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0,
                          i5)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l5, "lower bound 0");
      __builtin_annotation(u5, "upper bound 0");
      __builtin_annotation(l4, "lower bound 1");
      __builtin_annotation(u4, "upper bound 1");
      __builtin_annotation(l3, "lower bound 2");
      __builtin_annotation(u3, "upper bound 2");
      __builtin_annotation(l2, "lower bound 3");
      __builtin_annotation(u2, "upper bound 3");
      __builtin_annotation(l1, "lower bound 4");
      __builtin_annotation(u1, "upper bound 4");
      __builtin_annotation(l0, "lower bound 5");
      __builtin_annotation(u0, "upper bound 5");

      for (IType i5 = l5; i5 < u5; ++i5) {
        KOKKOS_IMPL_LOOP_5R(func, IType, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0,
                               i5)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_LOOP_REDUX_5R(value, func, IType, l4, l3, l2, l1, l0, u4, u3,
                                u2, u1, u0, i5)
    }
  }
};

// Rank = 6 non tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType>
struct Loop_Type<StrAssumption, Backend, 6, IType, /*LayoutLeft*/ true, void> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_LOOP_5L(func, IType, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0,
                          i5)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l5, "lower bound 0");
      __builtin_annotation(u5, "upper bound 0");
      __builtin_annotation(l4, "lower bound 1");
      __builtin_annotation(u4, "upper bound 1");
      __builtin_annotation(l3, "lower bound 2");
      __builtin_annotation(u3, "upper bound 2");
      __builtin_annotation(l2, "lower bound 3");
      __builtin_annotation(u2, "upper bound 3");
      __builtin_annotation(l1, "lower bound 4");
      __builtin_annotation(u1, "upper bound 4");
      __builtin_annotation(l0, "lower bound 5");
      __builtin_annotation(u0, "upper bound 5");

      for (IType i5 = l5; i5 < u5; ++i5) {
        KOKKOS_IMPL_LOOP_5L(func, IType, l4, l3, l2, l1, l0, u4, u3, u2, u1, u0,
                               i5)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_LOOP_REDUX_5L(value, func, IType, l4, l3, l2, l1, l0, u4, u3,
                                u2, u1, u0, i5)
    }
  }
};

// Rank = 6 tagged Right layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 6, IType, /*LayoutRight*/ false,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_5R(Tagged(), func, IType, l4, l3, l2, l1, l0, u4,
                                 u3, u2, u1, u0, i5)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l5, "lower bound 0");
      __builtin_annotation(u5, "upper bound 0");
      __builtin_annotation(l4, "lower bound 1");
      __builtin_annotation(u4, "upper bound 1");
      __builtin_annotation(l3, "lower bound 2");
      __builtin_annotation(u3, "upper bound 2");
      __builtin_annotation(l2, "lower bound 3");
      __builtin_annotation(u2, "upper bound 3");
      __builtin_annotation(l1, "lower bound 4");
      __builtin_annotation(u1, "upper bound 4");
      __builtin_annotation(l0, "lower bound 5");
      __builtin_annotation(u0, "upper bound 5");

      for (IType i5 = l5; i5 < u5; ++i5) {
        KOKKOS_IMPL_TAGGED_LOOP_5R(Tagged(), func, IType, l4, l3, l2, l1, l0,
                                      u4, u3, u2, u1, u0, i5)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[0];
    const IType u5 = static_cast<IType>(upper[0]);
    const IType l4 = (IType)lower[1];
    const IType u4 = static_cast<IType>(upper[1]);
    const IType l3 = (IType)lower[2];
    const IType u3 = static_cast<IType>(upper[2]);
    const IType l2 = (IType)lower[3];
    const IType u2 = static_cast<IType>(upper[3]);
    const IType l1 = (IType)lower[4];
    const IType u1 = static_cast<IType>(upper[4]);
    const IType l0 = (IType)lower[5];
    const IType u0 = static_cast<IType>(upper[5]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(Tagged(), value, func, IType, l4, l3, l2,
                                       l1, l0, u4, u3, u2, u1, u0, i5)
    }
  }
};

// Rank = 6 tagged Left layout
template <StringAssumption StrAssumption, StringAssumption Backend,
          typename IType, typename Tagged>
struct Loop_Type<StrAssumption, Backend, 6, IType, /*LayoutLeft*/ true,
                 Tagged> {
  template <typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_5L(Tagged(), func, IType, l4, l3, l2, l1, l0, u4,
                                 u3, u2, u1, u0, i5)
    }
  }

  template <typename Func, typename LoopBoundType>
  __attribute__((noinline)) static auto getApply(Func const& func,
                                                 const LoopBoundType& lower,
                                                 const LoopBoundType& upper) {
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    auto lambda    = [=]() -> void {
      __builtin_annotation((intptr_t)Backend.value, "backend");
      __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
      __builtin_annotation(l5, "lower bound 0");
      __builtin_annotation(u5, "upper bound 0");
      __builtin_annotation(l4, "lower bound 1");
      __builtin_annotation(u4, "upper bound 1");
      __builtin_annotation(l3, "lower bound 2");
      __builtin_annotation(u3, "upper bound 2");
      __builtin_annotation(l2, "lower bound 3");
      __builtin_annotation(u2, "upper bound 3");
      __builtin_annotation(l1, "lower bound 4");
      __builtin_annotation(u1, "upper bound 4");
      __builtin_annotation(l0, "lower bound 5");
      __builtin_annotation(u0, "upper bound 5");

      for (IType i5 = l5; i5 < u5; ++i5) {
        KOKKOS_IMPL_TAGGED_LOOP_5L(Tagged(), func, IType, l4, l3, l2, l1, l0,
                                      u4, u3, u2, u1, u0, i5)
      }
    };
    return lambda;
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  __attribute__((noinline, annotate("findscop"))) static void apply(
      ValType& value, Func const& func, const LoopBoundType& lower,
      const LoopBoundType& upper) {
    __builtin_annotation((intptr_t)Backend.value, "backend");
    __builtin_annotation((intptr_t)StrAssumption.value, "assumption");
    const IType l5 = (IType)lower[5];
    const IType u5 = static_cast<IType>(upper[5]);
    const IType l4 = (IType)lower[4];
    const IType u4 = static_cast<IType>(upper[4]);
    const IType l3 = (IType)lower[3];
    const IType u3 = static_cast<IType>(upper[3]);
    const IType l2 = (IType)lower[2];
    const IType u2 = static_cast<IType>(upper[2]);
    const IType l1 = (IType)lower[1];
    const IType u1 = static_cast<IType>(upper[1]);
    const IType l0 = (IType)lower[0];
    const IType u0 = static_cast<IType>(upper[0]);
    __builtin_annotation(l5, "lower bound 0");
    __builtin_annotation(u5, "upper bound 0");
    __builtin_annotation(l4, "lower bound 1");
    __builtin_annotation(u4, "upper bound 1");
    __builtin_annotation(l3, "lower bound 2");
    __builtin_annotation(u3, "upper bound 2");
    __builtin_annotation(l2, "lower bound 3");
    __builtin_annotation(u2, "upper bound 3");
    __builtin_annotation(l1, "lower bound 4");
    __builtin_annotation(u1, "upper bound 4");
    __builtin_annotation(l0, "lower bound 5");
    __builtin_annotation(u0, "upper bound 5");

    for (IType i5 = l5; i5 < u5; ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(Tagged(), value, func, IType, l4, l3, l2,
                                       l1, l0, u4, u3, u2, u1, u0, i5)
    }
  }
};

// end Structs for calling loops

template <StringAssumption StrAssumption, StringAssumption Backend, typename RP,
          typename Functor, typename Tag = void, typename ValueType = void,
          typename Enable = void>
struct HostIterate;

// For ParallelFor
template <StringAssumption StrAssumption, StringAssumption Backend, typename RP,
          typename Functor, typename Tag, typename ValueType>
struct HostIterate<StrAssumption, Backend, RP, Functor, Tag, ValueType,
                   std::enable_if_t<std::is_void<ValueType>::value>> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  inline HostIterate(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void operator()() const {
    // std::cout << "HostIterate ParallelFor" << std::endl;
    Loop_Type<StrAssumption, Backend, RP::rank, index_type, false, Tag>::apply(
        m_func, m_rp.m_lower, m_rp.m_upper);
  }

  auto getHostIterateFunction(/*RP const& rp, Functor const& func*/) const {
    return Loop_Type<StrAssumption, Backend, RP::rank, index_type, false,
                     Tag>::getApply(m_func, m_rp.m_lower, m_rp.m_upper);
  }

  RP const m_rp;
  Functor const m_func;
  std::conditional_t<std::is_void<Tag>::value, int, Tag> m_tag;
};

// For ParallelReduce
// ValueType - scalar: For reductions
template <StringAssumption StrAssumption, StringAssumption Backend, typename RP,
          typename Functor, typename Tag, typename ValueType>
struct HostIterate<StrAssumption, Backend, RP, Functor, Tag, ValueType,
                   std::enable_if_t<!std::is_void<ValueType>::value &&
                                    !std::is_array<ValueType>::value>> {
  using index_type = typename RP::index_type;

  using value_type = ValueType;

  inline HostIterate(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void operator()(value_type& val) const {
    Loop_Type<StrAssumption, Backend, RP::rank, index_type,
              (RP::inner_direction == Iterate::Left),
              Tag>::apply(val, m_func.get_functor(), m_rp.m_lower,
                          m_rp.m_upper);
  }

  RP const m_rp;
  Functor const m_func;
};

// For ParallelReduce
// Extra specialization for array reductions
// ValueType[]: For array reductions
template <StringAssumption StrAssumption, StringAssumption Backend, typename RP,
          typename Functor, typename Tag, typename ValueType>
struct HostIterate<StrAssumption, Backend, RP, Functor, Tag, ValueType,
                   std::enable_if_t<!std::is_void<ValueType>::value &&
                                    std::is_array<ValueType>::value>> {
  using index_type = typename RP::index_type;

  using value_type =
      std::remove_extent_t<ValueType>;  // strip away the
                                        // 'array-ness' [], only
                                        // underlying type remains

  inline HostIterate(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void operator()(value_type& val) const {
    Loop_Type<StrAssumption, Backend, RP::rank, index_type,
              (RP::inner_direction == Iterate::Left),
              Tag>::apply(val, m_func.get_functor(), m_rp.m_lower,
                          m_rp.m_upper);
  }

  RP const m_rp;
  Functor const m_func;
};

// ------------------------------------------------------------------

/* Functor call */
#undef KOKKOS_IMPL_APPLY
#undef KOKKOS_IMPL_APPLY_REDUX
#undef KOKKOS_IMPL_TAGGED_APPLY
#undef KOKKOS_IMPL_TAGGED_APPLY_REDUX

/* ParallelFor right */
#undef KOKKOS_IMPL_LOOP_1R
#undef KOKKOS_IMPL_LOOP_2R
#undef KOKKOS_IMPL_LOOP_3R
#undef KOKKOS_IMPL_LOOP_4R
#undef KOKKOS_IMPL_LOOP_5R
#undef KOKKOS_IMPL_LOOP_6R
#undef KOKKOS_IMPL_LOOP_7R

/* ParallelReduce right */
#undef KOKKOS_IMPL_LOOP_REDUX_1R
#undef KOKKOS_IMPL_LOOP_REDUX_2R
#undef KOKKOS_IMPL_LOOP_REDUX_3R
#undef KOKKOS_IMPL_LOOP_REDUX_4R
#undef KOKKOS_IMPL_LOOP_REDUX_5R
#undef KOKKOS_IMPL_LOOP_REDUX_6R
#undef KOKKOS_IMPL_LOOP_REDUX_7R

/* ParallelFor left */
#undef KOKKOS_IMPL_LOOP_1L
#undef KOKKOS_IMPL_LOOP_2L
#undef KOKKOS_IMPL_LOOP_3L
#undef KOKKOS_IMPL_LOOP_4L
#undef KOKKOS_IMPL_LOOP_5L
#undef KOKKOS_IMPL_LOOP_6L
#undef KOKKOS_IMPL_LOOP_7L

/* ParallelReduce left */
#undef KOKKOS_IMPL_LOOP_REDUX_1L
#undef KOKKOS_IMPL_LOOP_REDUX_2L
#undef KOKKOS_IMPL_LOOP_REDUX_3L
#undef KOKKOS_IMPL_LOOP_REDUX_4L
#undef KOKKOS_IMPL_LOOP_REDUX_5L
#undef KOKKOS_IMPL_LOOP_REDUX_6L
#undef KOKKOS_IMPL_LOOP_REDUX_7L

/* Tagged ParallelFor right */
#undef KOKKOS_IMPL_TAGGED_LOOP_1R
#undef KOKKOS_IMPL_TAGGED_LOOP_2R
#undef KOKKOS_IMPL_TAGGED_LOOP_3R
#undef KOKKOS_IMPL_TAGGED_LOOP_4R
#undef KOKKOS_IMPL_TAGGED_LOOP_5R
#undef KOKKOS_IMPL_TAGGED_LOOP_6R
#undef KOKKOS_IMPL_TAGGED_LOOP_7R

/* Tagged ParallelReduce right */
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_7R

/* Tagged ParallelFor left */
#undef KOKKOS_IMPL_TAGGED_LOOP_1L
#undef KOKKOS_IMPL_TAGGED_LOOP_2L
#undef KOKKOS_IMPL_TAGGED_LOOP_3L
#undef KOKKOS_IMPL_TAGGED_LOOP_4L
#undef KOKKOS_IMPL_TAGGED_LOOP_5L
#undef KOKKOS_IMPL_TAGGED_LOOP_6L
#undef KOKKOS_IMPL_TAGGED_LOOP_7L

/* Tagged ParallelReduce left */
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_7L

}  // namespace Impl
}  // namespace Kokkos

#endif

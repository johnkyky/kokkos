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
#define KOKKOS_IMPL_LOOP_1R(func, type, lower, upper, index, ...)          \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                              \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]); \
       ++i0) {                                                             \
    KOKKOS_IMPL_APPLY(func, __VA_ARGS__ __VA_OPT__(, ) i0)                 \
  }

#define KOKKOS_IMPL_LOOP_2R(func, type, lower, upper, index, ...)          \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]); \
       ++i1) {                                                             \
    KOKKOS_IMPL_LOOP_1R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i1)                     \
  }

#define KOKKOS_IMPL_LOOP_3R(func, type, lower, upper, index, ...)          \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]); \
       ++i2) {                                                             \
    KOKKOS_IMPL_LOOP_2R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i2)                     \
  }

#define KOKKOS_IMPL_LOOP_4R(func, type, lower, upper, index, ...)          \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]); \
       ++i3) {                                                             \
    KOKKOS_IMPL_LOOP_3R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i3)                     \
  }

#define KOKKOS_IMPL_LOOP_5R(func, type, lower, upper, index, ...)          \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]); \
       ++i4) {                                                             \
    KOKKOS_IMPL_LOOP_4R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i4)                     \
  }

#define KOKKOS_IMPL_LOOP_6R(func, type, lower, upper, index, ...)          \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]); \
       ++i5) {                                                             \
    KOKKOS_IMPL_LOOP_5R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i5)                     \
  }

#define KOKKOS_IMPL_LOOP_7R(func, type, lower, upper, index, ...)          \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]); \
       ++i6) {                                                             \
    KOKKOS_IMPL_LOOP_6R(func, type, lower, upper, index + 1,               \
                        __VA_ARGS__ __VA_OPT__(, ) i6)                     \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_LOOP_REDUX_1R(val, func, type, lower, upper, index, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);   \
       ++i0) {                                                               \
    KOKKOS_IMPL_APPLY_REDUX(val, func, __VA_ARGS__ __VA_OPT__(, ) i0)        \
  }

#define KOKKOS_IMPL_LOOP_REDUX_2R(val, func, type, lower, upper, index, ...) \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);   \
       ++i1) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_1R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i1)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_3R(val, func, type, lower, upper, index, ...) \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);   \
       ++i2) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_2R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i2)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_4R(val, func, type, lower, upper, index, ...) \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);   \
       ++i3) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_3R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i3)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_5R(val, func, type, lower, upper, index, ...) \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);   \
       ++i4) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_4R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i4)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_6R(val, func, type, lower, upper, index, ...) \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);   \
       ++i5) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_5R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i5)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_7R(val, func, type, lower, upper, index, ...) \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);   \
       ++i6) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_6R(val, func, type, lower, upper, index + 1,      \
                              __VA_ARGS__ __VA_OPT__(, ) i6)                 \
  }

/* Left layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_LOOP_1L(func, type, lower, upper, index, ...)          \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                              \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]); \
       ++i0) {                                                             \
    KOKKOS_IMPL_APPLY(func, i0 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_2L(func, type, lower, upper, index, ...)          \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]); \
       ++i1) {                                                             \
    KOKKOS_IMPL_LOOP_1L(func, type, lower, upper, index - 1,               \
                        i1 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_3L(func, type, lower, upper, index, ...)          \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]); \
       ++i2) {                                                             \
    KOKKOS_IMPL_LOOP_2L(func, type, lower, upper, index - 1,               \
                        i2 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_4L(func, type, lower, upper, index, ...)          \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]); \
       ++i3) {                                                             \
    KOKKOS_IMPL_LOOP_3L(func, type, lower, upper, index - 1,               \
                        i3 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_5L(func, type, lower, upper, index, ...)          \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]); \
       ++i4) {                                                             \
    KOKKOS_IMPL_LOOP_4L(func, type, lower, upper, index - 1,               \
                        i4 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_6L(func, type, lower, upper, index, ...)          \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]); \
       ++i5) {                                                             \
    KOKKOS_IMPL_LOOP_5L(func, type, lower, upper, index - 1,               \
                        i5 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

#define KOKKOS_IMPL_LOOP_7L(func, type, lower, upper, index, ...)          \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]); \
       ++i6) {                                                             \
    KOKKOS_IMPL_LOOP_6L(func, type, lower, upper, index - 1,               \
                        i6 __VA_OPT__(, ) __VA_ARGS__)                     \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_LOOP_REDUX_1L(val, func, type, lower, upper, index, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);   \
       ++i0) {                                                               \
    KOKKOS_IMPL_APPLY_REDUX(val, func, i0 __VA_OPT__(, ) __VA_ARGS__)        \
  }

#define KOKKOS_IMPL_LOOP_REDUX_2L(val, func, type, lower, upper, index, ...) \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);   \
       ++i1) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_1L(val, func, type, lower, upper, index - 1,      \
                              i1 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_3L(val, func, type, lower, upper, index, ...) \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);   \
       ++i2) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_2L(val, func, type, lower, upper, index - 1,      \
                              i2 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_4L(val, func, type, lower, upper, index, ...) \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);   \
       ++i3) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_3L(val, func, type, lower, upper, index - 1,      \
                              i3 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_5L(val, func, type, lower, upper, index, ...) \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);   \
       ++i4) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_4L(val, func, type, lower, upper, index - 1,      \
                              i4 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_6L(val, func, type, lower, upper, index, ...) \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);   \
       ++i5) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_5L(val, func, type, lower, upper, index - 1,      \
                              i5 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_LOOP_REDUX_7L(val, func, type, lower, upper, index, ...) \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);   \
       ++i6) {                                                               \
    KOKKOS_IMPL_LOOP_REDUX_6L(val, func, type, lower, upper, index - 1,      \
                              i6 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

/* Tagged loop implementation */
#define KOKKOS_IMPL_TAGGED_APPLY(tag, func, ...) func(tag, __VA_ARGS__);
#define KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func, ...) \
  func(tag, __VA_ARGS__, val);

/* Right layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_TAGGED_LOOP_1R(tag, func, type, lower, upper, index, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                 \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);    \
       ++i0) {                                                                \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, __VA_ARGS__ __VA_OPT__(, ) i0)        \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_2R(tag, func, type, lower, upper, index, ...) \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);    \
       ++i1) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_1R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i1)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_3R(tag, func, type, lower, upper, index, ...) \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);    \
       ++i2) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_2R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i2)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_4R(tag, func, type, lower, upper, index, ...) \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);    \
       ++i3) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_3R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i3)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_5R(tag, func, type, lower, upper, index, ...) \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);    \
       ++i4) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_4R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i4)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_6R(tag, func, type, lower, upper, index, ...) \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);    \
       ++i5) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_5R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i5)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_7R(tag, func, type, lower, upper, index, ...) \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);    \
       ++i6) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_6R(tag, func, type, lower, upper, index + 1,      \
                               __VA_ARGS__ __VA_OPT__(, ) i6)                 \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(tag, val, func, type, lower, upper, \
                                         index, ...)                         \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);   \
       ++i0) {                                                               \
    KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func,                           \
                                   __VA_ARGS__ __VA_OPT__(, ) i0)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);     \
       ++i1) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i1) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);     \
       ++i2) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i2) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);     \
       ++i3) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i3) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);     \
       ++i4) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i4) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);     \
       ++i5) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i5) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_7R(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);     \
       ++i6) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R(tag, val, func, type, lower, upper,       \
                                     index + 1, __VA_ARGS__ __VA_OPT__(, ) i6) \
  }

/* Left layout loop implementation */
/* ParallelFor */
#define KOKKOS_IMPL_TAGGED_LOOP_1L(tag, func, type, lower, upper, index, ...) \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                 \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);    \
       ++i0) {                                                                \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, i0 __VA_OPT__(, ) __VA_ARGS__)        \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_2L(tag, func, type, lower, upper, index, ...) \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);    \
       ++i1) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_1L(tag, func, type, lower, upper, index - 1,      \
                               i1 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_3L(tag, func, type, lower, upper, index, ...) \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);    \
       ++i2) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_2L(tag, func, type, lower, upper, index - 1,      \
                               i2 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_4L(tag, func, type, lower, upper, index, ...) \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);    \
       ++i3) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_3L(tag, func, type, lower, upper, index - 1,      \
                               i3 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_5L(tag, func, type, lower, upper, index, ...) \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);    \
       ++i4) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_4L(tag, func, type, lower, upper, index - 1,      \
                               i4 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_6L(tag, func, type, lower, upper, index, ...) \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);    \
       ++i5) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_5L(tag, func, type, lower, upper, index - 1,      \
                               i5 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_7L(tag, func, type, lower, upper, index, ...) \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);    \
       ++i6) {                                                                \
    KOKKOS_IMPL_TAGGED_LOOP_6L(tag, func, type, lower, upper, index - 1,      \
                               i6 __VA_OPT__(, ) __VA_ARGS__)                 \
  }

/* ParallelReduce */
#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(tag, val, func, type, lower, upper, \
                                         index, ...)                         \
  KOKKOS_ENABLE_IVDEP_MDRANGE                                                \
  for (type i0 = (type)lower[index]; i0 < static_cast<type>(upper[index]);   \
       ++i0) {                                                               \
    KOKKOS_IMPL_TAGGED_APPLY_REDUX(tag, val, func,                           \
                                   i0 __VA_OPT__(, ) __VA_ARGS__)            \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i1 = (type)lower[index]; i1 < static_cast<type>(upper[index]);     \
       ++i1) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(tag, val, func, type, lower, upper,       \
                                     index - 1, i1 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i2 = (type)lower[index]; i2 < static_cast<type>(upper[index]);     \
       ++i2) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(tag, val, func, type, lower, upper,       \
                                     index - 1, i2 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i3 = (type)lower[index]; i3 < static_cast<type>(upper[index]);     \
       ++i3) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(tag, val, func, type, lower, upper,       \
                                     index - 1, i3 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i4 = (type)lower[index]; i4 < static_cast<type>(upper[index]);     \
       ++i4) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(tag, val, func, type, lower, upper,       \
                                     index - 1, i4 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i5 = (type)lower[index]; i5 < static_cast<type>(upper[index]);     \
       ++i5) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(tag, val, func, type, lower, upper,       \
                                     index - 1, i5 __VA_OPT__(, ) __VA_ARGS__) \
  }

#define KOKKOS_IMPL_TAGGED_LOOP_REDUX_7L(tag, val, func, type, lower, upper,   \
                                         index, ...)                           \
  for (type i6 = (type)lower[index]; i6 < static_cast<type>(upper[index]);     \
       ++i6) {                                                                 \
    KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L(tag, val, func, type, lower, upper,       \
                                     index - 1, i6 __VA_OPT__(, ) __VA_ARGS__) \
  }

// ------------------------------------------------------------------ //

/* Structs for calling loops */
template <int Rank, typename IType, bool IsLeft, typename Tagged,
          typename Enable = void>
struct Loop_Type;

// Rank = 1 non taggged
template <typename IType, bool IsLeft>
struct Loop_Type<1, IType, IsLeft, void, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i0 = (IType)lower[index]; i0 < static_cast<IType>(upper[index]);
         ++i0) {
      KOKKOS_IMPL_APPLY(func, i0)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 1" << std::endl;
    constexpr int index = 0;
    for (IType i0 = (IType)lower[index]; i0 < static_cast<IType>(upper[index]);
         ++i0) {
      KOKKOS_IMPL_APPLY_REDUX(value, func, i0)
    }
  }
};

// Rank = 1 tagged
template <typename IType, bool IsLeft, typename Tagged>
struct Loop_Type<1, IType, IsLeft, Tagged, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 1" << std::endl;
    constexpr int index = 0;
    for (IType i0 = (IType)lower[index]; i0 < static_cast<IType>(upper[index]);
         ++i0) {
      KOKKOS_IMPL_TAGGED_APPLY(Tagged(), func, i0)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag left Rank 1" << std::endl;
    constexpr int index = 0;
    for (IType i0 = (IType)lower[index]; i0 < static_cast<IType>(upper[index]);
         ++i0) {
      KOKKOS_IMPL_TAGGED_APPLY_REDUX(Tagged(), value, func, i0)
    }
  }
};

// Rank = 2 non tagged Right layout
template <typename IType>
struct Loop_Type<2, IType, /*LayoutRight*/ false, void, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_1R(func, IType, lower, upper, index + 1, i1)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    constexpr int index = 0;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_REDUX_1R(value, func, IType, lower, upper, index + 1, i1)
    }
  }
};

// Rank = 2 non tagged Left layout
template <typename IType>
struct Loop_Type<2, IType, /*LayoutLeft*/ true, void, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    constexpr int index = 1;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_1L(func, IType, lower, upper, index - 1, i1)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    constexpr int index = 1;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_LOOP_REDUX_1L(value, func, IType, lower, upper, index - 1, i1)
    }
  }
};

// Rank = 2 tagged Right layout
template <typename IType, typename Tagged>
struct Loop_Type<2, IType, /*LayoutRight*/ false, Tagged, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 2 right" << std::endl;
    constexpr int index = 0;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_1R(Tagged(), func, IType, lower, upper, index + 1,
                                 i1)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 2 right" << std::endl;
    constexpr int index = 0;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R(Tagged(), value, func, IType, lower,
                                       upper, index + 1, i1)
    }
  }
};

// Rank = 2 tagged Left layout
template <typename IType, typename Tagged>
struct Loop_Type<2, IType, /*LayoutLeft*/ true, Tagged, void> {
  /* ParallelFor */
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 2 left" << std::endl;
    constexpr int index = 1;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_1L(Tagged(), func, IType, lower, upper, index - 1,
                                 i1)
    }
  }

  /* ParallelReduce */
  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 2 left" << std::endl;
    constexpr int index = 1;
    for (IType i1 = (IType)lower[index]; i1 < static_cast<IType>(upper[index]);
         ++i1) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L(Tagged(), value, func, IType, lower,
                                       upper, index - 1, i1)
    }
  }
};

// Rank = 3 non tagged Right layout
template <typename IType>
struct Loop_Type<3, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 3 right" << std::endl;
    constexpr int index = 0;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_2R(func, IType, lower, upper, index + 1, i2)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 3 right" << std::endl;
    constexpr int index = 0;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_REDUX_2R(value, func, IType, lower, upper, index + 1, i2)
    }
  }
};

// Rank = 3 non tagged Left layout
template <typename IType>
struct Loop_Type<3, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 3 left" << std::endl;
    constexpr int index = 2;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_2L(func, IType, lower, upper, index - 1, i2)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 3 left" << std::endl;
    constexpr int index = 2;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_LOOP_REDUX_2L(value, func, IType, lower, upper, index - 1, i2)
    }
  }
};

// Rank = 3 tagged Right layout
template <typename IType, typename Tagged>
struct Loop_Type<3, IType, /*LayoutRight*/ false, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 3 right" << std::endl;
    constexpr int index = 0;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_2R(Tagged(), func, IType, lower, upper, index + 1,
                                 i2)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 3 right" << std::endl;
    constexpr int index = 0;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R(Tagged(), value, func, IType, lower,
                                       upper, index + 1, i2)
    }
  }
};

// Rank = 3 tagged Left layout
template <typename IType, typename Tagged>
struct Loop_Type<3, IType, /*LayoutLeft*/ true, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 3 left" << std::endl;
    constexpr int index = 2;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_2L(Tagged(), func, IType, lower, upper, index - 1,
                                 i2)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 3 left" << std::endl;
    constexpr int index = 2;
    for (IType i2 = (IType)lower[index]; i2 < static_cast<IType>(upper[index]);
         ++i2) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L(Tagged(), value, func, IType, lower,
                                       upper, index - 1, i2)
    }
  }
};

// Rank = 4 non tagged Right layout
template <typename IType>
struct Loop_Type<4, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 4 right" << std::endl;
    constexpr int index = 0;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_LOOP_3R(func, IType, lower, upper, index + 1, i3)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 4 right" << std::endl;
    constexpr int index = 0;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_LOOP_REDUX_3R(value, func, IType, lower, upper, index + 1, i3)
    }
  }
};

// Rank = 4 non tagged Left layout
template <typename IType>
struct Loop_Type<4, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 4 left" << std::endl;
    constexpr int index = 3;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_LOOP_3L(func, IType, lower, upper, index - 1, i3)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 4 left" << std::endl;
    constexpr int index = 3;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_LOOP_REDUX_3L(value, func, IType, lower, upper, index - 1, i3)
    }
  }
};

// Rank = 4 tagged Right layout
template <typename IType, typename Tagged>
struct Loop_Type<4, IType, /*LayoutRight*/ false, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 4 right" << std::endl;
    constexpr int index = 0;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_3R(Tagged(), func, IType, lower, upper, index + 1,
                                 i3)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 4 right" << std::endl;
    constexpr int index = 0;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R(Tagged(), value, func, IType, lower,
                                       upper, index + 1, i3)
    }
  }
};

// Rank = 4 tagged Left layout
template <typename IType, typename Tagged>
struct Loop_Type<4, IType, /*LayoutLeft*/ true, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 4 left" << std::endl;
    constexpr int index = 3;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_3L(Tagged(), func, IType, lower, upper, index - 1,
                                 i3)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 4 left" << std::endl;
    constexpr int index = 3;
    for (IType i3 = (IType)lower[index]; i3 < static_cast<IType>(upper[index]);
         ++i3) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L(Tagged(), value, func, IType, lower,
                                       upper, index - 1, i3)
    }
  }
};

// Rank = 5 non tagged Right layout
template <typename IType>
struct Loop_Type<5, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 5 right" << std::endl;
    constexpr int index = 0;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_LOOP_4R(func, IType, lower, upper, index + 1, i4)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 5 right" << std::endl;
    constexpr int index = 0;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_LOOP_REDUX_4R(value, func, IType, lower, upper, index + 1, i4)
    }
  }
};

// Rank = 5 non tagged Left layout
template <typename IType>
struct Loop_Type<5, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 5 left" << std::endl;
    constexpr int index = 4;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_LOOP_4L(func, IType, lower, upper, index - 1, i4)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 5 left" << std::endl;
    constexpr int index = 4;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_LOOP_REDUX_4L(value, func, IType, lower, upper, index - 1, i4)
    }
  }
};

// Rank = 5 tagged Right layout
template <typename IType, typename Tagged>
struct Loop_Type<5, IType, /*LayoutRight*/ false, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 5 right" << std::endl;
    constexpr int index = 0;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_4R(Tagged(), func, IType, lower, upper, index + 1,
                                 i4)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 5 right" << std::endl;
    constexpr int index = 0;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R(Tagged(), value, func, IType, lower,
                                       upper, index + 1, i4)
    }
  }
};

// Rank = 5 tagged Left layout
template <typename IType, typename Tagged>
struct Loop_Type<5, IType, /*LayoutLeft*/ true, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 5 left" << std::endl;
    constexpr int index = 4;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_4L(Tagged(), func, IType, lower, upper, index - 1,
                                 i4)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 5 left" << std::endl;
    constexpr int index = 4;
    for (IType i4 = (IType)lower[index]; i4 < static_cast<IType>(upper[index]);
         ++i4) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L(Tagged(), value, func, IType, lower,
                                       upper, index - 1, i4)
    }
  }
};

// Rank = 6 non tagged Right layout
template <typename IType>
struct Loop_Type<6, IType, /*LayoutRight*/ false, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 6 right" << std::endl;
    constexpr int index = 0;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_LOOP_5R(func, IType, lower, upper, index + 1, i5)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 6 right" << std::endl;
    constexpr int index = 0;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_LOOP_REDUX_5R(value, func, IType, lower, upper, index + 1, i5)
    }
  }
};

// Rank = 6 non tagged Left layout
template <typename IType>
struct Loop_Type<6, IType, /*LayoutLeft*/ true, void, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "para Rank 6 left" << std::endl;
    constexpr int index = 5;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_LOOP_5L(func, IType, lower, upper, index - 1, i5)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "Rank 6 left" << std::endl;
    constexpr int index = 5;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_LOOP_REDUX_5L(value, func, IType, lower, upper, index - 1, i5)
    }
  }
};

// Rank = 6 tagged Right layout
template <typename IType, typename Tagged>
struct Loop_Type<6, IType, /*LayoutRight*/ false, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 6 right" << std::endl;
    constexpr int index = 0;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_5R(Tagged(), func, IType, lower, upper, index + 1,
                                 i5)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 6 right" << std::endl;
    constexpr int index = 0;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R(Tagged(), value, func, IType, lower,
                                       upper, index + 1, i5)
    }
  }
};

// Rank = 6 tagged Left layout
template <typename IType, typename Tagged>
struct Loop_Type<6, IType, /*LayoutLeft*/ true, Tagged, void> {
  template <typename Func, typename LoopBoundType>
  static void apply(Func const& func, const LoopBoundType& lower,
                    const LoopBoundType& upper) {
    std::cout << "parallel tag Rank 6 left" << std::endl;
    constexpr int index = 5;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_5L(Tagged(), func, IType, lower, upper, index - 1,
                                 i5)
    }
  }

  template <typename ValType, typename Func, typename LoopBoundType>
  static void apply(ValType& value, Func const& func,
                    const LoopBoundType& lower, const LoopBoundType& upper) {
    std::cout << "reduce tag Rank 6 left" << std::endl;
    constexpr int index = 5;
    for (IType i5 = (IType)lower[index]; i5 < static_cast<IType>(upper[index]);
         ++i5) {
      KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L(Tagged(), value, func, IType, lower,
                                       upper, index - 1, i5)
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

  inline HostIterate(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void operator()() const {
    std::cout << "HostIterate ParallelFor" << std::endl;
    Loop_Type<RP::rank, index_type, (RP::inner_direction == Iterate::Left),
              Tag>::apply(m_func, m_rp.m_lower, m_rp.m_upper);
  }

  RP const m_rp;
  Functor const m_func;
  std::conditional_t<std::is_void<Tag>::value, int, Tag> m_tag;
};

// For ParallelReduce
// ValueType - scalar: For reductions
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterate<RP, Functor, Tag, ValueType,
                   std::enable_if_t<!std::is_void<ValueType>::value &&
                                    !std::is_array<ValueType>::value>> {
  using index_type = typename RP::index_type;

  using value_type = ValueType;

  inline HostIterate(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline void operator()(value_type& val) const {
    Loop_Type<RP::rank, index_type, (RP::inner_direction == Iterate::Left),
              Tag>::apply(val, m_func.get_functor(), m_rp.m_lower,
                          m_rp.m_upper);
  }

  RP const m_rp;
  Functor const m_func;
};

// For ParallelReduce
// Extra specialization for array reductions
// ValueType[]: For array reductions
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct HostIterate<RP, Functor, Tag, ValueType,
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
    Loop_Type<RP::rank, index_type, (RP::inner_direction == Iterate::Left),
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
#undef KOKKOS_IMPL_LOOP_8R

/* ParallelReduce right */
#undef KOKKOS_IMPL_LOOP_REDUX_1R
#undef KOKKOS_IMPL_LOOP_REDUX_2R
#undef KOKKOS_IMPL_LOOP_REDUX_3R
#undef KOKKOS_IMPL_LOOP_REDUX_4R
#undef KOKKOS_IMPL_LOOP_REDUX_5R
#undef KOKKOS_IMPL_LOOP_REDUX_6R
#undef KOKKOS_IMPL_LOOP_REDUX_7R
#undef KOKKOS_IMPL_LOOP_REDUX_8R

/* ParallelFor left */
#undef KOKKOS_IMPL_LOOP_1L
#undef KOKKOS_IMPL_LOOP_2L
#undef KOKKOS_IMPL_LOOP_3L
#undef KOKKOS_IMPL_LOOP_4L
#undef KOKKOS_IMPL_LOOP_5L
#undef KOKKOS_IMPL_LOOP_6L
#undef KOKKOS_IMPL_LOOP_7L
#undef KOKKOS_IMPL_LOOP_8L

/* ParallelReduce left */
#undef KOKKOS_IMPL_LOOP_REDUX_1L
#undef KOKKOS_IMPL_LOOP_REDUX_2L
#undef KOKKOS_IMPL_LOOP_REDUX_3L
#undef KOKKOS_IMPL_LOOP_REDUX_4L
#undef KOKKOS_IMPL_LOOP_REDUX_5L
#undef KOKKOS_IMPL_LOOP_REDUX_6L
#undef KOKKOS_IMPL_LOOP_REDUX_7L
#undef KOKKOS_IMPL_LOOP_REDUX_8L

/* Tagged ParallelFor right */
#undef KOKKOS_IMPL_TAGGED_LOOP_1R
#undef KOKKOS_IMPL_TAGGED_LOOP_2R
#undef KOKKOS_IMPL_TAGGED_LOOP_3R
#undef KOKKOS_IMPL_TAGGED_LOOP_4R
#undef KOKKOS_IMPL_TAGGED_LOOP_5R
#undef KOKKOS_IMPL_TAGGED_LOOP_6R
#undef KOKKOS_IMPL_TAGGED_LOOP_7R
#undef KOKKOS_IMPL_TAGGED_LOOP_8R

/* Tagged ParallelReduce right */
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_1R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_2R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_3R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_4R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_5R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_6R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_7R
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_8R

/* Tagged ParallelFor left */
#undef KOKKOS_IMPL_TAGGED_LOOP_1L
#undef KOKKOS_IMPL_TAGGED_LOOP_2L
#undef KOKKOS_IMPL_TAGGED_LOOP_3L
#undef KOKKOS_IMPL_TAGGED_LOOP_4L
#undef KOKKOS_IMPL_TAGGED_LOOP_5L
#undef KOKKOS_IMPL_TAGGED_LOOP_6L
#undef KOKKOS_IMPL_TAGGED_LOOP_7L
#undef KOKKOS_IMPL_TAGGED_LOOP_8L

/* Tagged ParallelReduce left */
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_1L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_2L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_3L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_4L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_5L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_6L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_7L
#undef KOKKOS_IMPL_TAGGED_LOOP_REDUX_8L

}  // namespace Impl
}  // namespace Kokkos

#endif

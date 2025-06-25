// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file memory.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/memory/memory.hpp"
#include "shamalgs/details/memory/avoidCopyMemory.hpp"
#include "shamalgs/details/memory/fallbackMemory.hpp"

namespace shamalgs::memory {

    template<class T>
    T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx) {
        return details::AvoidCopy<T>::extract_element(q, buf, idx);
    }

    template<class T>
    sycl::buffer<T> vec_to_buf(const std::vector<T> &vec) {
        return details::Fallback<T>::vec_to_buf(vec);
    }

    template<class T>
    std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len) {
        return details::Fallback<T>::buf_to_vec(buf, len);
    }

#define XMAC_TYPES                                                                                 \
    X(f32)                                                                                         \
    X(f32_2)                                                                                       \
    X(f32_3)                                                                                       \
    X(f32_4)                                                                                       \
    X(f32_8)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64)                                                                                         \
    X(f64_2)                                                                                       \
    X(f64_3)                                                                                       \
    X(f64_4)                                                                                       \
    X(f64_8)                                                                                       \
    X(f64_16)                                                                                      \
    X(u8)                                                                                          \
    X(u32)                                                                                         \
    X(u32_2)                                                                                       \
    X(u32_3)                                                                                       \
    X(u32_4)                                                                                       \
    X(u32_8)                                                                                       \
    X(u32_16)                                                                                      \
    X(u64)                                                                                         \
    X(u64_2)                                                                                       \
    X(u64_3)                                                                                       \
    X(u64_4)                                                                                       \
    X(u64_8)                                                                                       \
    X(u64_16)                                                                                      \
    X(i64_3)                                                                                       \
    X(i64)

#define X(_arg_)                                                                                   \
    template _arg_ extract_element(sycl::queue &q, sycl::buffer<_arg_> &buf, u32 idx);             \
    template sycl::buffer<_arg_> vec_to_buf(const std::vector<_arg_> &buf);                        \
    template std::vector<_arg_> buf_to_vec(sycl::buffer<_arg_> &buf, u32 len);

    XMAC_TYPES
#undef X

} // namespace shamalgs::memory

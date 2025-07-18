// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file fallbackMemory.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/memory/fallbackMemory.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace shamalgs::memory::details {

    template<class T>
    T Fallback<T>::extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx) {

        T ret_val;
        {
            sycl::host_accessor acc{buf, sycl::read_only};
            ret_val = acc[idx];
        }

        return ret_val;
    }

    template<class T>
    sycl::buffer<T> Fallback<T>::vec_to_buf(const std::vector<T> &vec) {
        sycl::buffer<T> ret(vec.size());

        {
            sycl::host_accessor acc{ret, sycl::write_only, sycl::no_init};

            for (u32 idx = 0; idx < vec.size(); idx++) {
                acc[idx] = vec[idx];
            }
        }

        return std::move(ret);
    }

    template<class T>
    std::vector<T> Fallback<T>::buf_to_vec(sycl::buffer<T> &buf, u32 len) {
        std::vector<T> ret;
        ret.resize(len);

        {
            sycl::host_accessor acc{buf, sycl::read_only};

            for (u32 idx = 0; idx < len; idx++) {
                ret[idx] = acc[idx];
            }
        }

        return std::move(ret);
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

#define X(_arg_) template struct Fallback<_arg_>;
    XMAC_TYPES
#undef X

} // namespace shamalgs::memory::details

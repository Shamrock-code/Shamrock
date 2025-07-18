// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file avoidCopyMemory.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/memory/avoidCopyMemory.hpp"
#include "shambase/exception.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace shamalgs::memory::details {

    template<class T>
    T AvoidCopy<T>::extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx) {

        if (!(idx < buf.size())) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "you are trying to access out of bounds");
        }

        sycl::buffer<T> len_value{1};
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor global_mem{buf, cgh, sycl::read_only};
            sycl::accessor acc_rec{len_value, cgh, sycl::write_only, sycl::no_init};

            u32 idx_ = idx;

            cgh.single_task([=]() {
                acc_rec[0] = global_mem[idx_];
            });
        });

        T ret_val;
        {
            sycl::host_accessor acc{len_value, sycl::read_only};
            ret_val = acc[0];
        }

        return ret_val;
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

#define X(_arg_) template struct AvoidCopy<_arg_>;
    XMAC_TYPES
#undef X

} // namespace shamalgs::memory::details

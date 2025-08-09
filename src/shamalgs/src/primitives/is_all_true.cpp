// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file is_all_true.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements functions to check if all elements in a buffer are non-zero (true).
 */

#include "shamalgs/primitives/is_all_true.hpp"

template<class T>
bool shamalgs::primitives::is_all_true(sham::DeviceBuffer<T> &buf, u32 cnt) {

    // TODO do it on GPU pleeeaze
    {
        auto tmp = buf.copy_to_stdvec();

        for (u32 i = 0; i < cnt; i++) {
            if (tmp[i] == 0) {
                return false;
            }
        }
    }

    return true;
}

template<class T>
bool shamalgs::primitives::is_all_true(sycl::buffer<T> &buf, u32 cnt) {

    // TODO do it on GPU pleeeaze
    {
        sycl::host_accessor acc{buf, sycl::read_only};

        for (u32 i = 0; i < cnt; i++) {
            if (acc[i] == 0) {
                return false;
            }
        }
    }

    return true;
}

template bool shamalgs::primitives::is_all_true(sycl::buffer<u8> &buf, u32 cnt);
template bool shamalgs::primitives::is_all_true(sham::DeviceBuffer<u8> &buf, u32 cnt);

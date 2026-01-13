// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file fallbackMemory.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"

namespace shamalgs::memory::details {

    template<class T>
    struct Fallback {

        static T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

        static sycl::buffer<T> vec_to_buf(const std::vector<T> &vec);
        static std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);
    };

} // namespace shamalgs::memory::details

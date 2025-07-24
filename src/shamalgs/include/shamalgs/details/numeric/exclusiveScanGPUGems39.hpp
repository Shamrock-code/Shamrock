// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file exclusiveScanGPUGems39.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_1(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_2(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_3(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

} // namespace shamalgs::numeric::details

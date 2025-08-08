// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file is_all_true.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs {

    template<class T>
    bool is_all_true(sycl::buffer<T> &buf, u32 cnt);

    template<class T>
    bool is_all_true(sham::DeviceBuffer<T> &buf, u32 cnt);

} // namespace shamalgs

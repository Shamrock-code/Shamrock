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
 * @file scan_exclusive_sum_in_place.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Provides an in-place exclusive scan (prefix sum) operation on a device buffer.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

    template<class T>
    void scan_exclusive_sum_in_place(sham::DeviceBuffer<T> &buf1, u32 len);

}

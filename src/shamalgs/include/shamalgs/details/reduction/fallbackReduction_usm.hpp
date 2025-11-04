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
 * @file fallbackReduction_usm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    T sum_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    template<class T>
    T min_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    template<class T>
    T max_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

} // namespace shamalgs::reduction::details

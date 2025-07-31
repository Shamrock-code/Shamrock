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
 * @file groupReduction_usm.hpp
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

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

namespace shamalgs::reduction::details {

    template<class T>
    T sum_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);
    template<class T>
    T min_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);
    template<class T>
    T max_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);

} // namespace shamalgs::reduction::details

#endif

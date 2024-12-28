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
 * @file groupReduction_usm.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

#define SHAMALGS_GROUP_REDUCTION_SUPPORT

// so far Acpp does not support group reduction with generic backend
#ifdef __HIPSYCL_ENABLE_LLVM_SSCP_TARGET__
    #undef SHAMALGS_GROUP_REDUCTION_SUPPORT
#endif

#ifdef SHAMALGS_GROUP_REDUCTION_SUPPORT

namespace shamalgs::reduction::details {

    template<class T>
    T sum_usm_group(
        sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);
    template<class T>
    T min_usm_group(
        sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);
    template<class T>
    T max_usm_group(
        sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size);

} // namespace shamalgs::reduction::details

#endif

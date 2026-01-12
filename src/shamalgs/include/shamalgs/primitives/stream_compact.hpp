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
 * @file sort_by_keys.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

    sham::DeviceBuffer<u32> stream_compact(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> && buf_flags, u32 len);

} // namespace shamalgs::primitives

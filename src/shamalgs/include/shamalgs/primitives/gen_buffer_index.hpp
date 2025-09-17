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
 * @file gen_buffer_index.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

    /// generate a buffer where buf[i] = i
    sham::DeviceBuffer<u32> gen_buffer_index(sham::DeviceScheduler_ptr sched, u32 len);

    /// fill a buffer where buf[i] = i
    void fill_buffer_index(sham::DeviceBuffer<u32> &buf, u32 len);
} // namespace shamalgs::primitives

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
 * @brief Provides functions to generate and fill buffers with sequential indices.
 *
 */

#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Generates a buffer where buf[i] = i.
     *
     * @param sched The scheduler to run on.
     * @param len Length of the buffer to generate.
     * @return sham::DeviceBuffer<u32> The generated buffer containing sequential indices.
     */
    sham::DeviceBuffer<u32> gen_buffer_index(sham::DeviceScheduler_ptr sched, u32 len);

    /**
     * @brief Fills a buffer with sequential indices, such that buf[i] = i.
     *
     * The buffer must have a size of at least `len`. If `len` is 0, the function has no effect.
     *
     * @param buf The buffer to fill.
     * @param len The number of elements to fill from the start of the buffer.
     * @throws std::invalid_argument if buf.get_size() < len.
     */
    void fill_buffer_index(sham::DeviceBuffer<u32> &buf, u32 len);

} // namespace shamalgs::primitives

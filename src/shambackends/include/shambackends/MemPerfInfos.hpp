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
 * @file MemPerfInfos.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include <cstddef>

namespace sham {

    /**
     * @brief Structure to store the performance informations about memory allocation and
     * deallocation
     *
     * This structure contains the times and allocated bytes for each memory space.
     */
    struct MemPerfInfos {
        /// Time spent allocating memory on the host
        f64 time_alloc_host = 0;
        /// Time spent allocating memory on the device
        f64 time_alloc_device = 0;
        /// Time spent allocating memory in shared memory
        f64 time_alloc_shared = 0;

        /// Time spent deallocating memory on the host
        f64 time_free_host = 0;
        /// Time spent deallocating memory on the device
        f64 time_free_device = 0;
        /// Time spent deallocating memory in shared memory
        f64 time_free_shared = 0;

        /// Bytes allocated on the host
        size_t allocated_byte_host = 0;
        /// Bytes allocated on the device
        size_t allocated_byte_device = 0;
        /// Bytes allocated in shared memory
        size_t allocated_byte_shared = 0;

        /// max bytes allocated on the host
        size_t max_allocated_byte_host = 0;
        /// max bytes allocated on the device
        size_t max_allocated_byte_device = 0;
        /// max bytes allocated in shared memory
        size_t max_allocated_byte_shared = 0;
    };

} // namespace sham

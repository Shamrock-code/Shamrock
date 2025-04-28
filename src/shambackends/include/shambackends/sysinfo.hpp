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
 * @file sysinfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <cstddef>
#include <optional>

namespace sham {

    /**
     * @brief Get the amount of physical memory (RAM) available on the system, in bytes.
     *
     * @return The amount of physical memory available, or std::nullopt if the information
     *         cannot be retrieved.
     *
     * @details This function is implemented for Mac OS X and Linux. Other platforms will
     *          return std::nullopt.
     */
    std::optional<std::size_t> getPhysicalMemory();
} // namespace sham

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
 * @file timestep_report.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <string>

namespace shammodels {

    std::string report_perf_timestep(
        f64 rate,
        u64 nobj,
        u64 npatch,
        f64 tcompute,
        f64 mpi_timer,
        f64 alloc_time_device,
        f64 alloc_time_host,
        size_t max_mem_device,
        size_t max_mem_host);

} // namespace shammodels

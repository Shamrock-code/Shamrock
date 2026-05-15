// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file format_time.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Human-readable nanosecond duration formatting
 *
 */

#include "shambase/string.hpp"

namespace shambase {

    /**
     * @brief Convert nanoseconds to a human-readable string representation.
     *
     * @param nanosec The duration in nanoseconds.
     * @return std::string The duration in a human-readable format.
     */
    inline std::string nanosec_to_time_str(double nanosec) {
        double sec_int = nanosec;

        std::string unit = "ns";

        if (sec_int > 2000) {
            unit = "us";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "ms";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "s";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "ks";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "Ms";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "Gs";
            sec_int /= 1000;
        }

        return shambase::format("{:.2f} {}", sec_int, unit);
    }
} // namespace shambase

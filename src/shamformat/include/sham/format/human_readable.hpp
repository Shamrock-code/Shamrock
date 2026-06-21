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
 * @file human_readable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Convert raw numeric values to human-readable SI-formatted pairs
 */

#include <array>
#include <cmath>
#include <utility>

namespace sham {

    namespace details {

        /**
         * @brief Generate a compile-time table of SI prefixes and their magnitudes.
         *
         * @tparam allow_below_1 When true, includes sub-unity prefixes (nano through
         * milli). When false, starts at unity (no sub-unity prefixes).
         * @return constexpr std::array of {prefix, magnitude} pairs sorted ascending
         */
        template<bool allow_below_1>
        consteval auto make_si_pairs() {
            if constexpr (allow_below_1) {
                return std::array<std::pair<const char *, double>, 12>{
                    {{"n", 1e-9},
                     {"u", 1e-6},
                     {"m", 1e-3},
                     {"", 1.0},
                     {"k", 1e3},
                     {"M", 1e6},
                     {"G", 1e9},
                     {"T", 1e12},
                     {"P", 1e15},
                     {"E", 1e18},
                     {"Z", 1e21},
                     {"Y", 1e24}}};
            } else {
                return std::array<std::pair<const char *, double>, 9>{
                    {{"", 1.0},
                     {"k", 1e3},
                     {"M", 1e6},
                     {"G", 1e9},
                     {"T", 1e12},
                     {"P", 1e15},
                     {"E", 1e18},
                     {"Z", 1e21},
                     {"Y", 1e24}}};
            }
        }

    } // namespace details

    /**
     * @brief Struct holding a scaled value with its SI prefix
     *
     * @param value the scaled numeric value (e.g. 1.5 for "1.5 k")
     * @param prefix the SI prefix character (e.g. "k", "M", "")
     * @param ratio the ratio used to scale the original value
     */
    struct human_readable_t {
        double value;
        const char *prefix;
        double ratio;
    };

    /**
     * @brief Convert a raw value to a human-readable scaled form with an SI prefix.
     *
     * Finds the largest SI prefix whose magnitude divides evenly into `value`,
     * returning the scaled value, prefix character, and division ratio. Values
     * are clamped to the smallest or largest available SI unit when they fall
     * outside the supported range. Zero always returns an empty prefix.
     *
     * @tparam allow_below_1 When true (default), the full table including nano/
     * micro/milli is used. When false, only prefixes >= 1 are considered.
     * @param value the raw numeric value to scale
     * @return human_readable_t the scaled value with prefix information
     */
    template<bool allow_below_1 = true>
    inline human_readable_t to_human_readable(double value) {
        static constexpr auto si = details::make_si_pairs<allow_below_1>();

        double ax = std::fabs(value);

        // zero: no prefix
        if (ax == 0.0) {
            return {.value = 0.0, .prefix = "", .ratio = 1.0};
        }

        // too large, clamp to largest unit
        const auto &largest = si.back();
        if (ax >= largest.second) {
            return {
                .value = value / largest.second, .prefix = largest.first, .ratio = largest.second};
        }

        for (int i = static_cast<int>(si.size()) - 2; i >= 0; --i) {
            if (ax >= si[i].second) {
                return {
                    .value = value / si[i].second, .prefix = si[i].first, .ratio = si[i].second};
            }
        }

        // too small, clamp to smallest unit
        const auto &smallest = si.front();
        return {
            .value = value / smallest.second, .prefix = smallest.first, .ratio = smallest.second};
    }

} // namespace sham

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
 * @file json_print_diff.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>

namespace shamrock {

    inline std::string json_diff_str(const nlohmann::json &j1, const nlohmann::json &j2) {
        auto v1 = shambase::split_str(j1.dump(4), "\n");
        auto v2 = shambase::split_str(j2.dump(4), "\n");

        // Equivalent to diff since the json entries are sorted
        auto it1 = v1.begin();
        auto it2 = v2.begin();

        auto green = shambase::term_colors::col8b_green();
        auto red   = shambase::term_colors::col8b_red();
        auto reset = shambase::term_colors::reset();

        // Helper lambda to check if two strings differ only by a trailing character
        auto differ_by_trailing_char = [](const std::string &s1, const std::string &s2) -> bool {
            if (s1 == s2)
                return false;
            size_t len1 = s1.length();
            size_t len2 = s2.length();

            // Check if s1 equals s2 with last char removed
            if (len1 == len2 + 1 && s1.substr(0, len2) == s2)
                return true;

            // Check if s2 equals s1 with last char removed
            if (len2 == len1 + 1 && s2.substr(0, len1) == s1)
                return true;

            return false;
        };

        std::stringstream ss;

        while (it1 != v1.end() || it2 != v2.end()) {
            if (it1 == v1.end()) {
                ss << shambase::format("{}+ | {}{}\n", green, *it2++, reset);
            } else if (it2 == v2.end()) {
                ss << shambase::format("{}- | {}{}\n", red, *it1++, reset);
            } else if (differ_by_trailing_char(*it1, *it2)) {
                auto &longest = (*it1).length() > (*it2).length() ? *it1 : *it2;
                ss << shambase::format("  | {}\n", longest); // unchanged
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ss << shambase::format("{}- | {}{}\n", red, *it1++, reset);
            } else if (*it2 < *it1) {
                ss << shambase::format("{}+ | {}{}\n", green, *it2++, reset);
            } else {
                ss << shambase::format("  | {}\n", *it1); // unchanged
                ++it1;
                ++it2;
            }
        }

        return ss.str();
    }
} // namespace shamrock

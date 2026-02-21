// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file json_utils.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/io/json_print_diff.hpp"
#include "shamrock/io/json_utils.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>

namespace shamrock {

    std::string log_json_changes(
        const nlohmann::json &j_current,
        const nlohmann::json &j,
        bool has_used_defaults,
        bool has_updated_config) {

        int dash_count = 50;

        std::string faint_line = shambase::format(
            "{}{:-^{}}{}\n",
            shambase::term_colors::faint(),
            "",         // empty string
            dash_count, // dynamic width
            shambase::term_colors::reset());

        auto green = shambase::term_colors::col8b_green();
        auto reset = shambase::term_colors::reset();
        auto red   = shambase::term_colors::col8b_red();

        std::string newold_text
            = shambase::format("[{}+ | added{}] [{}- | removed{}]", green, reset, red, reset);
        std::string nice_plus  = shambase::format("{}+{}", green, reset);
        std::string nice_minus = shambase::format("{}-{}", red, reset);

        std::string log = shambase::format(
            "Used config parameters are listed below;\n      "
            "Possible cases for highlighted entries:\n      "
            "  - ({})   default-added\n      "
            "  - ({}/{}) updated \n      "
            "  - ({})   ignored values \n      "
            "List of changes: {}\n",
            nice_plus,
            nice_plus,
            nice_minus,
            nice_minus,
            newold_text);

        std::stringstream ss;
        ss << log;

        ss << faint_line;
        ss << shamrock::json_diff_str(j, j_current);
        ss << faint_line;

        return ss.str();
    }

} // namespace shamrock

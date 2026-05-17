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
 * @file color_env.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <source_location>
#include <string_view>
#include <functional>
#include <optional>
#include <stdexcept>

namespace sham::term {

    struct TermSupportEnvVars {
        std::optional<std::string_view> TERM;
        std::optional<std::string_view> COLORTERM;
        std::optional<std::string_view> NO_COLOR;
        std::optional<std::string_view> CLICOLOR_FORCE;
    };

    using term_parse_callback_t
        = std::function<std::runtime_error(const char *what, std::source_location where)>;

    // TODO is runtime_error the right excepotion ?
    void parse_terminal_support(TermSupportEnvVars vars, term_parse_callback_t error_callback);

} // namespace sham::term

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
 * @file env.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Environment variable parsing for terminal color and size configuration (TERM, COLORTERM, NO_COLOR, CLICOLOR_FORCE, SHAMTTYCOL)
 *
 */

#include "sham/term/error_callback.hpp"
#include <string_view>
#include <optional>

namespace sham::term {

    /// @brief Holds optional terminal environment variables (TERM, COLORTERM, NO_COLOR, CLICOLOR_FORCE, SHAMTTYCOL)
    struct TermEnvVars {
        std::optional<std::string_view> TERM;
        std::optional<std::string_view> COLORTERM;
        std::optional<std::string_view> NO_COLOR;
        std::optional<std::string_view> CLICOLOR_FORCE;
        std::optional<std::string_view> COLUMN;
    };

    /// @brief Parses terminal environment variables to determine color support and set terminal size
    void parse_terminal_support(TermEnvVars vars, const term_parse_callback_t &error_callback);

} // namespace sham::term

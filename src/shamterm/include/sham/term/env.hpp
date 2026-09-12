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
 * @brief Environment variable parsing for terminal color, size and UTF-8 support configuration
 * (TERM, COLORTERM, NO_COLOR, CLICOLOR_FORCE, COLUMN, LANG, LC_ALL, LC_CTYPE, NO_UTF8,
 * FORCE_UTF8)
 *
 */

#include "sham/term/error_callback.hpp"
#include <string_view>
#include <optional>

namespace sham::term {

    /// @brief Holds optional terminal environment variables (TERM, COLORTERM, NO_COLOR,
    /// CLICOLOR_FORCE, COLUMN, LANG, LC_ALL, LC_CTYPE, NO_UTF8, FORCE_UTF8)
    ///
    /// Note: the LC_ALL/LC_CTYPE env vars are exposed as lc_all/lc_ctype here since LC_ALL and
    /// LC_CTYPE are reserved macro names defined by <locale.h>.
    struct TermEnvVars {
        std::optional<std::string_view> TERM;
        std::optional<std::string_view> COLORTERM;
        std::optional<std::string_view> NO_COLOR;
        std::optional<std::string_view> CLICOLOR_FORCE;
        std::optional<std::string_view> COLUMN;
        std::optional<std::string_view> LANG;
        std::optional<std::string_view> lc_all;
        std::optional<std::string_view> lc_ctype;
        std::optional<std::string_view> NO_UTF8;
        std::optional<std::string_view> FORCE_UTF8;
    };

    /// @brief Parses terminal environment variables to determine color support and set terminal
    /// size
    void parse_terminal_support(TermEnvVars vars, const term_parse_callback_t &error_callback);

} // namespace sham::term

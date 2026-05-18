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
 * @file tty_env.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "sham/term/error_callback.hpp"
#include <string_view>
#include <optional>

namespace sham::term {

    /// Holds terminal environment variables collected from the environment.
    struct TtyEnvVars {
        /// The SHAMTTYCOL environment variable to set the terminal width (tty columns).
        std::optional<std::string_view> SHAMTTYCOL;
    };

    /// Parses terminal environment variables, invoking error_callback on invalid input.
    void parse_tty_env_vars(TtyEnvVars vars, const term_parse_callback_t &error_callback);

} // namespace sham::term

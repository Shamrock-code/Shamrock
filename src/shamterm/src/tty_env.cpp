// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file tty_env.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "sham/term/tty_env.hpp"
#include "sham/term/tty.hpp"
#include <source_location>
#include <string>

namespace sham::term {

    void parse_tty_env_vars(TtyEnvVars vars, const term_parse_callback_t &error_callback) {

        auto &res = vars.SHAMTTYCOL;

        int min_sz = 10;
        if (res) {
            try {
                int val = std::stoi(std::string(*res));
                if (val < min_sz) {
                    val = min_sz;
                }
                sham::term::set_tty_columns(val);
            } catch (const std::invalid_argument &a) {
                throw error_callback(
                    "Error : SHAMTTYCOL is not an integer", std::source_location::current());
            } catch (const std::out_of_range &a) {
                throw error_callback(
                    "Error : SHAMTTYCOL is out of range", std::source_location::current());
            }
        }
    }

} // namespace sham::term

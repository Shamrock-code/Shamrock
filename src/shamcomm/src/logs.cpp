// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file logs.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/logs/reformat_message.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <cmath>

namespace shamcomm::logs {

    ///////////////////////////////////
    // log level declared printer
    ///////////////////////////////////

    /// X macro impl for the print_active_level() function
#define IsActivePrint(_name, StructREF) _name##_ln("xxx", "xxx", "(", "logger::" #_name, ")");

    void print_active_level() {
        raw_ln("log status : ");
        if (get_loglevel() == i8_max) {
            raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        raw_ln(shambase::format(" - Loglevel: {}, enabled log types :", u32(get_loglevel())));

// logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);

/// Temp definition for the X macro call in print_active_level()
#define X IsActivePrint
        LIST_LEVEL
#undef X
        // logger::raw_ln(terminal_effects::faint + "----------------------" +
        // terminal_effects::reset);
    }

#undef IsActivePrint

} // namespace shamcomm::logs

std::string LogLevel_DebugAlloc::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}

std::string LogLevel_DebugMPI::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_blue(), level_name, module_name, in);
}

std::string LogLevel_DebugSYCL::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(
        shambase::term_colors::col8b_magenta(), level_name, module_name, in);
}

std::string LogLevel_Debug::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_green(), level_name, module_name, in);
}

std::string LogLevel_Info::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_cyan(), "Info", module_name, in);
}

std::string LogLevel_Normal::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_simple(shambase::term_colors::empty(), level_name, module_name, in);
}

std::string LogLevel_Warning::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_yellow(), level_name, module_name, in);
}

std::string LogLevel_Error::reformat(const std::string &in, std::string module_name) {
    return logger::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}

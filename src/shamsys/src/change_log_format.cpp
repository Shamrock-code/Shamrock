// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file change_log_format.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/term_colors.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/tty.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"

/**
 * @brief Namespace for log formatters
 */
namespace logformatter {

    /**
     * @brief Log formatter for style 0, full details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */

    std::string style0_formatter_full(const logger::ReformatArgs &args) {
        return "[" + (args.color) + args.module_name + shambase::term_colors::reset() + "] "
               + (args.color) + (args.level_name) + shambase::term_colors::reset() + ": "
               + args.content;
    }

    /**
     * @brief Log formatter for style 0, simple details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style0_formatter_simple(const logger::ReformatArgs &args) {
        return "[" + (args.color) + args.module_name + shambase::term_colors::reset() + "] "
               + (args.color) + (args.level_name) + shambase::term_colors::reset() + ": "
               + args.content;
    }

    /**
     * @brief Log formatter for style 1, full details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style1_formatter_full(const logger::ReformatArgs &args) {
        return shambase::format(
            "{5:}rank={6:<4}{2:} {5:}({3:^20}){2:} {0:}{1:}{2:}: {4:}",
            args.color,
            args.level_name,
            shambase::term_colors::reset(),
            args.module_name,
            args.content,
            shambase::term_colors::faint(),
            shamcomm::world_rank());
    }

    /**
     * @brief Log formatter for style 1, simple details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style1_formatter_simple(const logger::ReformatArgs &args) {
        return shambase::format(
            "{5:}({3:}){2:} : {4:}",
            args.color,
            args.level_name,
            shambase::term_colors::reset(),
            args.module_name,
            args.content,
            shambase::term_colors::faint());
    }
    /**
     * @brief Log formatter for style 2, full details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */

    std::string style2_formatter_full(const logger::ReformatArgs &args) {

        return shambase::format(
            "{0:}{1:}{2:}: {4:}{5:} | ({3:}) rank={6:<4}{2:}",
            args.color,
            args.level_name,
            shambase::term_colors::reset(),
            args.module_name,
            args.content,
            shambase::term_colors::faint(),
            shamcomm::world_rank());
    }

    /**
     * @brief Log formatter for style 2, simple details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style2_formatter_simple(const logger::ReformatArgs &args) {
        return shambase::format(
            "{5:}({3:}){2:} : {4:}",
            args.color,
            args.level_name,
            shambase::term_colors::reset(),
            args.module_name,
            args.content,
            shambase::term_colors::faint());
    }

    /**
     * @brief Log formatter for style 3, full details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style3_formatter_full(const logger::ReformatArgs &args) {

        u32 tty_width = shamcmdopt::get_tty_columns();

        std::string ansi_reset = shambase::term_colors::reset();
        std::string ansi_faint = shambase::term_colors::faint();

        std::string lineend = shambase::format(
            "{5:} [{3:}][rank={6:}]{2:}",
            args.color,
            args.level_name,
            ansi_reset,
            args.module_name,
            args.content,
            ansi_faint,
            shamcomm::world_rank());

        std::string log = shambase::format(
            "{0:}{1:}{2:}: {4:}",
            args.color,
            args.level_name,
            ansi_reset,
            args.module_name,
            args.content,
            ansi_faint,
            shamcomm::world_rank());

        std::string log_line1, log_line2;
        size_t first_nl = log.find_first_of('\n');
        if (first_nl != std::string::npos) {
            log_line1 = log.substr(0, first_nl);
            log_line2 = log.substr(first_nl);
        } else {
            log_line1 = log;
            log_line2 = "";
        }

        u32 ansi_count = ansi_reset.size() * 2 + ansi_faint.size() + args.color.size();

        return shambase::format("{:<{}}", log_line1, tty_width - lineend.size() + ansi_count - 1)
               + lineend + log_line2;
    }

    /**
     * @brief Log formatter for style 3, simple details
     *
     * @param args The arguments for the log formatter
     * @return std::string The formatted log
     */
    std::string style3_formatter_simple(const logger::ReformatArgs &args) {
        return shambase::format(
            "{5:}{3:}{2:}: {4:}",
            args.color,
            args.level_name,
            shambase::term_colors::reset(),
            args.module_name,
            args.content,
            shambase::term_colors::bold());
    }

    /**
     * @brief The callback called when an exception is thrown
     *
     * This callback is called with the formatted exception message as argument.
     * It is settable with set_exception_gen_callback.
     *
     * @param msg The formatted exception message
     */
    void exception_gen_callback(std::string msg) {
        shamcomm::logs::err_ln("Exception", "Exception created :\n" + msg);
    }

} // namespace logformatter

std::string SHAMLOGFORMATTER = shamcmdopt::getenv_str_default_register(
    "SHAMLOGFORMATTER", "3", "Change the log formatter (values :0-3) [default: 3]");

std::string SHAMLOG_ERR_ON_EXCEPT = shamcmdopt::getenv_str_default_register(
    "SHAMLOG_ERR_ON_EXCEPT", "1", "Enable logging of exceptions (default to 1)");

namespace shamsys {

    void change_log_format() {

        shamlog_debug_ln("Sys", "changing formatter to MPI form");

        if (SHAMLOGFORMATTER == "0") {
            logger::change_formaters(
                logformatter::style0_formatter_full, logformatter::style0_formatter_simple);
        } else if (SHAMLOGFORMATTER == "1") {
            logger::change_formaters(
                logformatter::style1_formatter_full, logformatter::style1_formatter_simple);
        } else if (SHAMLOGFORMATTER == "2") {
            logger::change_formaters(
                logformatter::style2_formatter_full, logformatter::style2_formatter_simple);
        } else if (SHAMLOGFORMATTER == "3") {
            logger::change_formaters(
                logformatter::style3_formatter_full, logformatter::style3_formatter_simple);
        } else {
            logger::err_ln("Log", "Unknown formatter");
            shambase::throw_unimplemented("Unknown formatter");
        }

        if (SHAMLOG_ERR_ON_EXCEPT == "1") {
            shamlog_debug_ln("Log", "Enabling exception handler callback");
            shambase::set_exception_gen_callback(&logformatter::exception_gen_callback);
        }
    }
} // namespace shamsys

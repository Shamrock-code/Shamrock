// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file logs.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <cmath>

namespace shamcomm::logs {

    /**
     * @brief Pointer to the full formatter function.
     *
     * If this pointer is equal to nullptr, the default full formatter function
     * will be used.
     */
    reformat_func_ptr _reformat_all = nullptr;

    /**
     * @brief Pointer to the simple formatter function.
     *
     * If this pointer is equal to nullptr, the default simple formatter function
     * will be used.
     */
    reformat_func_ptr _reformat_simple = nullptr;

    void change_formaters(reformat_func_ptr full, reformat_func_ptr simple) {
        _reformat_simple = simple;
        _reformat_all    = full;
    }

    /**
     * @brief Format a log message with all the information
     * @param color The color of the log message
     * @param name The name of the logger
     * @param module_name The name of the module
     * @param content The content of the log message
     * @return A formatted log message
     */
    inline std::string reformat_all(
        std::string color, const char *name, std::string module_name, std::string content) {
        if (shamcomm::logs::_reformat_all == nullptr) {
            // old form
            return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
                   + (name) + shambase::term_colors::reset() + ": " + content;
        }

        return shamcomm::logs::_reformat_all({color, name, module_name, content});
    }

    /**
     * @brief Format a log message with the minimum information
     * @param color The color of the log message
     * @param name The name of the logger
     * @param module_name The name of the module
     * @param content The content of the log message
     * @return A formatted log message
     */
    inline std::string reformat_simple(
        std::string color, const char *name, std::string module_name, std::string content) {

        if (shamcomm::logs::_reformat_simple == nullptr) {
            // old form
            return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
                   + (name) + shambase::term_colors::reset() + ": " + content;
        }

        return shamcomm::logs::_reformat_simple({color, name, module_name, content});
    }

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

    ///////////////////////////////////
    // Code init done
    ///////////////////////////////////

    void code_init_done_log() {

        auto lines = std::array<std::string, 16>{
            // Someone that coded too much here
            "Now it's time to " + shambase::term_colors::col8b_cyan()
                + shambase::term_colors::blink() + "ROCK" + shambase::term_colors::reset() + ".",
            "Shamrock rolls — no time for moss.", // Rolling stone gathers no moss.
            "Shamrock's live — go with the flow.",
            "Shamrock — as solid as a rock.",
            "Shamrock's stable and steady as a rock.",
            "Shamrock initialized — no cracks in this rock.",
            "Shamrock is ready to eat cheese (melted) and bread.",
            "Are you sure you want to work today?",
            "No holidays for the Shamrock ... (yeah, this was a PhD at some point)",
            "-[--->+<]>--.>+[----->+++<]>+.-------.++++++++++++.+++++.---.------------.++++++++.",
            "CPU hours to burn? We don't do such thing here.",
            "Are you burning GPUs or CPUs today?",

            // Someone that started on oumuamua
            "Shamrock your way to a brighter day!",
            "Node hours to burn? Leaf it to me.",
            "Ready for some shamazing simulations?",
            R"=(
While you wait for this simulation to run, give that cat a hug!

    |\__/,|   (`\
  _.|o o  |_   ) )
-(((---(((--------
            )="};

        auto get_sentence = [&]() {
            f64 t   = shambase::details::get_wtime();
            u64 idx = static_cast<u64>(std::floor(
                          t * 2503'09713 // you wont guess what this stands for
                          ))
                      % lines.size();
            return lines[idx];
        };

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
            logger::raw_ln(
                " - Code init:",
                shambase::term_colors::col8b_green() + "DONE" + shambase::term_colors::reset()
                    + ".",
                get_sentence());
            logger::print_faint_row();
        }
    }

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

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file reformat_message.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/logs/reformat_message.hpp"
#include "shambase/term_colors.hpp"

namespace shambase::logs {

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
    std::string reformat_all(
        const std::string &color,
        const char *name,
        const std::string &module_name,
        const std::string &content) {
        if (shambase::logs::_reformat_all == nullptr) {
            // old form
            return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
                   + (name) + shambase::term_colors::reset() + ": " + content;
        }

        return shambase::logs::_reformat_all({color, name, module_name, content});
    }

    /**
     * @brief Format a log message with the minimum information
     * @param color The color of the log message
     * @param name The name of the logger
     * @param module_name The name of the module
     * @param content The content of the log message
     * @return A formatted log message
     */
    std::string reformat_simple(
        const std::string &color,
        const char *name,
        const std::string &module_name,
        const std::string &content) {

        if (shambase::logs::_reformat_simple == nullptr) {
            // old form
            return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
                   + (name) + shambase::term_colors::reset() + ": " + content;
        }

        return shambase::logs::_reformat_simple({color, name, module_name, content});
    }
} // namespace shambase::logs

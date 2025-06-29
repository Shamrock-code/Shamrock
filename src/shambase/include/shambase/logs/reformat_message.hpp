// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file reformat_message.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <string>
namespace shambase::logs {

    /**
     * @brief A structure containing the arguments to a log formatter
     *
     * This structure is used to pass the arguments to a log formatter
     * function. It contains the color of the log message, the name of the
     * log level, the name of the module, and the log message itself.
     */
    struct ReformatArgs {
        /**
         * @brief The color of the log message
         *
         * This is a string containing the color escape sequence for the
         * log message. It may be empty if no color is desired.
         */
        std::string color;

        /**
         * @brief The name of the log level
         *
         * This is a string containing the name of the log level. It may
         * be empty if no log level is desired.
         */
        const char *level_name;

        /**
         * @brief The name of the module from which the log is emitter
         *
         * This is a string containing the name of the module. It may be
         * empty if no module name is desired.
         */
        std::string module_name;

        /**
         * @brief The log message
         *
         * This is a string containing the log message. It may be empty if
         * no message is desired.
         */
        std::string content;
    };

    /**
     * @brief A pointer to a log formatter function
     *
     * This is a pointer to a function that takes a ReformatArgs structure
     * and returns a string containing the formatted log message. It is
     * used to pass a log formatter function to the change_formaters
     * function.
     */
    using reformat_func_ptr = std::string (*)(const ReformatArgs &args);

    /**
     * @brief Changes the log formatter functions
     *
     * This function is used to change the log formatter functions used by
     * the shamcomm::logs module. It takes two reformat_func_ptr
     * arguments: the first is the full formatter function, and the second
     * is the simple formatter function.
     *
     * @param full The full formatter function
     * @param simple The simple formatter function
     */
    void change_formaters(reformat_func_ptr full, reformat_func_ptr simple);

    /**
     * @brief Format a log message with all the information
     * @param color The color of the log message
     * @param name The name of the logger
     * @param module_name The name of the module
     * @param content The content of the log message
     * @return A formatted log message
     */
    std::string
    reformat_all(std::string color, const char *name, std::string module_name, std::string content);

    /**
     * @brief Format a log message with the minimum information
     * @param color The color of the log message
     * @param name The name of the logger
     * @param module_name The name of the module
     * @param content The content of the log message
     * @return A formatted log message
     */
    std::string reformat_simple(
        std::string color, const char *name, std::string module_name, std::string content);

} // namespace shambase::logs

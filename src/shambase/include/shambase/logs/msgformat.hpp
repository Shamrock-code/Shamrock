// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file msgformat.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"

namespace shambase::logs {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log message formatting
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Formats an empty log message
     *
     * This function is used to format log messages when there are no additional
     * arguments to be passed.
     *
     * @return An empty string
     */
    inline std::string format_message() { return ""; }

    /**
     * Formats a log message with multiple arguments.
     *
     * @param var1 The first argument to be formatted in the log message.
     * @param var2 The remaining arguments to be formatted in the log message.
     *
     * @return The formatted log message.
     *
     */
    template<typename T, typename... Types>
    std::string format_message(T var1, Types... var2);

    /**
     * Formats a log message by concatenating a string with additional arguments.
     *
     * @param s The initial string to be formatted
     * @param var2 Additional arguments to be formatted and appended to the string
     *
     * @return The formatted log message
     */
    template<typename... Types>
    inline std::string format_message(std::string s, Types... var2) {
        return s + " " + format_message(var2...);
    }

    /**
     * @brief Formats a log message with multiple arguments.
     *
     * This function is a variadic template that recursively formats a log message
     * by concatenating the string representation of each argument passed to it.
     *
     * @tparam T The type of the first argument
     * @tparam Types The types of the remaining arguments
     *
     * @return The formatted log message.
     *
     */
    template<typename T, typename... Types>
    inline std::string format_message(T var1, Types... var2) {
        // Special case for string literals
        if constexpr (std::is_same_v<T, const char *>) {
            // Convert the string literal to a std::string and concatenate it with the formatted
            // string from the remaining arguments
            return std::string(var1) + " " + format_message(var2...);
        }
        // Special case for pointer types
        else if constexpr (std::is_pointer_v<T>) {
            // Convert the pointer to a void pointer, format it as a hexadecimal string, and
            // concatenate it with the formatted string from the remaining arguments
            return shambase::format("{} ", static_cast<const void *>(var1))
                   + format_message(var2...);
        }

        else {
            // General case for other types
            // Format the argument as a string and concatenate it with the formatted string from the
            // remaining arguments
            return shambase::format("{} ", var1) + format_message(var2...);
        }
    }

} // namespace shambase::logs

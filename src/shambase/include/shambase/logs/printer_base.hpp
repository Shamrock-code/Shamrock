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
 * @file printer_base.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/logs/msgformat.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"

namespace shambase::logs {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // log message printing
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Prints a log message with no arguments.
     *
     * This function is a specialization of the variadic template `print` that
     * does nothing. It is used to terminate the recursion when there are no
     * more arguments to format.
     */
    inline void print() {}

    /**
     * @brief Prints a log message with multiple arguments.
     *
     * This function is a variadic template that recursively prints a log message
     * by formatting each argument and concatenating them. It uses the
     * `shambase::print` function to output the formatted message.
     *
     * @tparam T The type of the first argument
     * @tparam Types The types of the remaining arguments
     *
     * @param var1 The first argument to be printed in the log message.
     * @param var2 The remaining arguments to be printed in the log message.
     */
    template<typename T, typename... Types>
    void print(T var1, Types... var2) {
        shambase::print(shambase::logs::format_message(var1, var2...));
    }

    /**
     * @brief Prints a log message with multiple arguments followed by a newline.
     *
     * This function is a specialization of the variadic template `print_ln` that
     * does nothing. It is used to terminate the recursion when there are no
     * more arguments to format.
     */
    inline void print_ln() {}

    /**
     * @brief Prints a log message with multiple arguments followed by a newline.
     *
     * This function is a variadic template that recursively prints a log message
     * by formatting each argument and concatenating them. It uses the
     * `shambase::println` function to output the formatted message, followed by
     * a newline.
     *
     * @tparam T The type of the first argument
     * @tparam Types The types of the remaining arguments
     *
     * @param var1 The first argument to be printed in the log message.
     * @param var2 The remaining arguments to be printed in the log message.
     */
    template<typename T, typename... Types>
    void print_ln(T var1, Types... var2) {
        shambase::println(shambase::logs::format_message(var1, var2...));
        shambase::flush();
    }

} // namespace shambase::logs

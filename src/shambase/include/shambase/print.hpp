// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file print.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <string>

namespace shambase {

    /**
     * @brief Prints a string to the console.
     *
     * @param s The string to be printed.
     */
    void print(std::string s);

    /**
     * @brief Prints a string to the console followed by a newline.
     *
     * @param s The string to be printed.
     */
    void println(std::string s);

    /**
     * @brief Flushes the output buffer.
     *
     * This function forces the output buffer to be written to the console
     * immediately, rather than waiting for the buffer to fill up.
     */
    void flush();

    /**
     * @brief Changes the behavior of the print, println and flush functions.
     *
     * @param func_printer_normal The function to be used for printing a string
     *                           to the console.
     * @param func_printer_ln The function to be used for printing a string to
     *                        the console followed by a newline.
     * @param func_flush_func The function to be used for flushing the output
     *                       buffer.
     */
    void change_printer(
        void (*func_printer_normal)(std::string),
        void (*func_printer_ln)(std::string),
        void (*func_flush_func)());

    /**
     * @brief Restores the default behavior of the print and println functions.
     *
     * This function resets the behavior of the print and println functions to
     * their default behavior, which is to print to the standard output stream.
     */
    void reset_std_behavior();

} // namespace shambase

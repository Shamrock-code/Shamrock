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
 * @file logs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/logs.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include <string>

/**
 * @brief X-macro for log level definition
 *
 * This X-macro is used to define the log levels available in the library. It is
 * used to generate the log level structs and the corresponding log formatter
 * functions.
 *
 * The X-macro takes the following form:
 *
 *   X(name, LogLevel_Name)
 *
 * Where:
 *
 *   - name is the name of the log level
 *   - LogLevel_Name is the name of the struct containing the log level
 *     information
 */
#define LIST_LEVEL                                                                                 \
    X(debug_alloc, LogLevel_DebugAlloc)                                                            \
    X(debug_mpi, LogLevel_DebugMPI)                                                                \
    X(debug_sycl, LogLevel_DebugSYCL)                                                              \
    X(debug, LogLevel_Debug)                                                                       \
    X(info, LogLevel_Info)                                                                         \
    X(normal, LogLevel_Normal)                                                                     \
    X(warn, LogLevel_Warning)                                                                      \
    X(err, LogLevel_Error)

namespace shamcomm::logs {

    using namespace shambase::logs;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Base print without decoration
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Prints a log message with multiple arguments without newline.
     *
     * This is a variadic template that recursively prints a log message
     * by formatting each argument and concatenating them. It uses the
     * `shambase::print` function to output the formatted message.
     *
     * @tparam Types The types of the arguments to be printed
     *
     * @param var2 The arguments to be printed
     */
    template<typename... Types>
    inline void raw(Types... var2) {
        print(var2...);
    }

    /**
     * @brief Prints a log message with multiple arguments followed by a newline.
     *
     * This is a variadic template that recursively prints a log message
     * by formatting each argument and concatenating them. It uses the
     * `shambase::print` function to output the formatted message, followed
     * by a newline.
     *
     * @tparam Types The types of the arguments to be printed
     *
     * @param var2 The arguments to be printed
     */
    template<typename... Types>
    inline void raw_ln(Types... var2) {
        print_ln(var2...);
    }

    /**
     * @brief Prints a faint separator line to the log.
     *
     * This is a convenience function to print a separator line to the log.
     * It prints a line of 50 dashes using the `shambase::term_colors::faint` color.
     */
    inline void print_faint_row() {
        raw_ln(
            shambase::term_colors::faint() + "-----------------------------------------------------"
            + shambase::term_colors::reset());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log levels
    ////////////////////////////////////////////////////////////////////////////////////////////////

/// Macro defining the log levels that will be expanded by an X-macro
#define DECLARE_LOG_LEVEL(_name, StructREF)                                                        \
                                                                                                   \
    constexpr i8 log_##_name = (StructREF::logval);                                                \
                                                                                                   \
    template<typename... Types>                                                                    \
    inline void _name(std::string module_name, Types... var2) {                                    \
        if (details::loglevel >= log_##_name) {                                                    \
            shamcomm::logs::print(                                                                 \
                StructREF::reformat(shamcomm::logs::format_message(var2...), module_name));        \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    template<typename... Types>                                                                    \
    inline void _name##_ln(std::string module_name, Types... var2) {                               \
        if (details::loglevel >= log_##_name) {                                                    \
            shamcomm::logs::print_ln(                                                              \
                StructREF::reformat(shamcomm::logs::format_message(var2...), module_name));        \
        }                                                                                          \
    }

/// Temp definition for the X macro call to define the log levels
#define X DECLARE_LOG_LEVEL
    LIST_LEVEL
#undef X

#undef DECLARE_LOG_LEVEL

    /**
     * @brief Prints the active log levels.
     */
    void print_active_level();

    /// Indicates that the code initialization is complete through various means ;)
    void code_init_done_log();

} // namespace shamcomm::logs

/**
 * @fn shamcomm::logs::debug_alloc(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::debug_alloc_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_debug_alloc
 * @brief the log level value associated with debug_alloc
 */

/**
 * @fn shamcomm::logs::debug_mpi(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::debug_mpi_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_debug_mpi
 * @brief the log level value associated with debug_mpi
 */

/**
 * @fn shamcomm::logs::debug_sycl(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::debug_sycl_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_debug_sycl
 * @brief the log level value associated with debug_sycl
 */

/**
 * @fn shamcomm::logs::debug(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::debug_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_debug
 * @brief the log level value associated with debug
 */

/**
 * @fn shamcomm::logs::info(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::info_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_info
 * @brief the log level value associated with info
 */

/**
 * @fn shamcomm::logs::normal(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::normal_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_normal
 * @brief the log level value associated with normal
 */

/**
 * @fn shamcomm::logs::warn(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::warn_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_warn
 * @brief the log level value associated with warn
 */

/**
 * @fn shamcomm::logs::err(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::err_ln(std::string module_name, Types... var2)
 * @brief Prints a log message with multiple arguments followed by a newline.
 *
 * @param module_name The name of the module
 * @param var2 The arguments to be printed
 */

/**
 * @fn shamcomm::logs::log_err
 * @brief the log level value associated with err
 */

/**
 * @brief alias namespace to simplify the use of log functions
 */
namespace logger {

    using namespace shamcomm::logs;
    using namespace shambase::logs;

} // namespace logger

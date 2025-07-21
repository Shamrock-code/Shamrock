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
 * @file loglevels.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include <string>

/**
 * @brief Log level struct for debugging memory allocation
 *
 * This struct contains the log level information for debugging memory allocation.
 */
struct LogLevel_DebugAlloc {
    constexpr static i8 logval              = 127;           ///< Log level value
    constexpr static const char *level_name = "Debug Alloc"; ///< Log level name

    /**
     * @brief Log formatter function for debugging memory allocation
     *
     * This function formats the log message for debugging memory allocation.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_debug_alloc(module_name, ...)                                                      \
    if (shambase::logs::details::loglevel >= LogLevel_DebugAlloc::logval) {                        \
        shambase::logs::print(LogLevel_DebugAlloc::reformat(                                       \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define shamlog_debug_alloc_ln(module_name, ...)                                                   \
    if (shambase::logs::details::loglevel >= LogLevel_DebugAlloc::logval) {                        \
        shambase::logs::print_ln(LogLevel_DebugAlloc::reformat(                                    \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define when_shamlog_debug_alloc                                                                   \
    if (shambase::logs::details::loglevel >= LogLevel_DebugAlloc::logval)

/**
 * @brief Log level struct for debugging MPI operations
 *
 * This struct contains the log level information for debugging MPI operations.
 */
struct LogLevel_DebugMPI {
    constexpr static i8 logval              = 100;         ///< Log level value
    constexpr static const char *level_name = "Debug MPI"; ///< Log level name

    /**
     * @brief Log formatter function for debugging MPI operations
     *
     * This function formats the log message for debugging MPI operations.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_debug_mpi(module_name, ...)                                                        \
    if (shambase::logs::details::loglevel >= LogLevel_DebugMPI::logval) {                          \
        shambase::logs::print(LogLevel_DebugMPI::reformat(                                         \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define shamlog_debug_mpi_ln(module_name, ...)                                                     \
    if (shambase::logs::details::loglevel >= LogLevel_DebugMPI::logval) {                          \
        shambase::logs::print_ln(LogLevel_DebugMPI::reformat(                                      \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define when_shamlog_debug_mpi if (shambase::logs::details::loglevel >= LogLevel_DebugMPI::logval)

/**
 * @brief Log level struct for debugging SYCL operations
 *
 * This struct contains the log level information for debugging SYCL operations.
 */
struct LogLevel_DebugSYCL {
    constexpr static i8 logval              = 11;           ///< Log level value
    constexpr static const char *level_name = "Debug SYCL"; ///< Log level name

    /**
     * @brief Log formatter function for debugging SYCL operations
     *
     * This function formats the log message for debugging SYCL operations.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_debug_sycl(module_name, ...)                                                       \
    if (shambase::logs::details::loglevel >= LogLevel_DebugSYCL::logval) {                         \
        shambase::logs::print(LogLevel_DebugSYCL::reformat(                                        \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define shamlog_debug_sycl_ln(module_name, ...)                                                    \
    if (shambase::logs::details::loglevel >= LogLevel_DebugSYCL::logval) {                         \
        shambase::logs::print_ln(LogLevel_DebugSYCL::reformat(                                     \
            shambase::logs::format_message(__VA_ARGS__), module_name));                            \
    }

#define when_shamlog_debug_sycl if (shambase::logs::details::loglevel >= LogLevel_DebugSYCL::logval)

/**
 * @brief Log level struct for debugging general operations
 *
 * This struct contains the log level information for debugging general operations.
 */
struct LogLevel_Debug {
    constexpr static i8 logval              = 10;      ///< Log level value
    constexpr static const char *level_name = "Debug"; ///< Log level name

    /**
     * @brief Log formatter function for debugging general operations
     *
     * This function formats the log message for debugging general operations.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_debug(module_name, ...)                                                            \
    if (shambase::logs::details::loglevel >= LogLevel_Debug::logval) {                             \
        shambase::logs::print(                                                                     \
            LogLevel_Debug::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));   \
    }

#define shamlog_debug_ln(module_name, ...)                                                         \
    if (shambase::logs::details::loglevel >= LogLevel_Debug::logval) {                             \
        shambase::logs::print_ln(                                                                  \
            LogLevel_Debug::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));   \
    }

#define when_shamlog_debug if (shambase::logs::details::loglevel >= LogLevel_Debug::logval)

/**
 * @brief Log level struct for informational messages
 *
 * This struct contains the log level information for informational messages.
 */
struct LogLevel_Info {
    constexpr static i8 logval              = 1;  ///< Log level value
    constexpr static const char *level_name = ""; ///< Log level name

    /**
     * @brief Log formatter function for informational messages
     *
     * This function formats the log message for informational messages.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_info(module_name, ...)                                                             \
    if (shambase::logs::details::loglevel >= LogLevel_Info::logval) {                              \
        shambase::logs::print(                                                                     \
            LogLevel_Info::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));    \
    }

#define shamlog_info_ln(module_name, ...)                                                          \
    if (shambase::logs::details::loglevel >= LogLevel_Info::logval) {                              \
        shambase::logs::print_ln(                                                                  \
            LogLevel_Info::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));    \
    }

#define when_shamlog_info if (shambase::logs::details::loglevel >= LogLevel_Info::logval)

/**
 * @brief Log level struct for normal messages
 *
 * This struct contains the log level information for normal messages.
 */
struct LogLevel_Normal {
    constexpr static i8 logval              = 0;  ///< Log level value
    constexpr static const char *level_name = ""; ///< Log level name

    /**
     * @brief Log formatter function for normal messages
     *
     * This function formats the log message for normal messages.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_normal(module_name, ...)                                                           \
    if (shambase::logs::details::loglevel >= LogLevel_Normal::logval) {                            \
        shambase::logs::print(                                                                     \
            LogLevel_Normal::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));  \
    }

#define shamlog_normal_ln(module_name, ...)                                                        \
    if (shambase::logs::details::loglevel >= LogLevel_Normal::logval) {                            \
        shambase::logs::print_ln(                                                                  \
            LogLevel_Normal::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));  \
    }

#define when_shamlog_normal if (shambase::logs::details::loglevel >= LogLevel_Normal::logval)

/**
 * @brief Log level struct for warning messages
 *
 * This struct contains the log level information for warning messages.
 */
struct LogLevel_Warning {
    constexpr static i8 logval              = -1;        ///< Log level value
    constexpr static const char *level_name = "Warning"; ///< Log level name

    /**
     * @brief Log formatter function for warning messages
     *
     * This function formats the log message for warning messages.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_warn(module_name, ...)                                                             \
    if (shambase::logs::details::loglevel >= LogLevel_Warning::logval) {                           \
        shambase::logs::print(                                                                     \
            LogLevel_Warning::reformat(shambase::logs::format_message(__VA_ARGS__), module_name)); \
    }

#define shamlog_warn_ln(module_name, ...)                                                          \
    if (shambase::logs::details::loglevel >= LogLevel_Warning::logval) {                           \
        shambase::logs::print_ln(                                                                  \
            LogLevel_Warning::reformat(shambase::logs::format_message(__VA_ARGS__), module_name)); \
    }

#define when_shamlog_warn if (shambase::logs::details::loglevel >= LogLevel_Warning::logval)

/**
 * @brief Log level struct for error messages
 *
 * This struct contains the log level information for error messages.
 */
struct LogLevel_Error {
    constexpr static i8 logval              = -10;     ///< Log level value
    constexpr static const char *level_name = "Error"; ///< Log level name

    /**
     * @brief Log formatter function for error messages
     *
     * This function formats the log message for error messages.
     *
     * @param in The log message
     * @param module_name The name of the module
     * @return The formatted log message
     */
    static std::string reformat(const std::string &in, std::string module_name);
};

#define shamlog_error(module_name, ...)                                                            \
    if (shambase::logs::details::loglevel >= LogLevel_Error::logval) {                             \
        shambase::logs::print(                                                                     \
            LogLevel_Error::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));   \
    }

#define shamlog_error_ln(module_name, ...)                                                         \
    if (shambase::logs::details::loglevel >= LogLevel_Error::logval) {                             \
        shambase::logs::print_ln(                                                                  \
            LogLevel_Error::reformat(shambase::logs::format_message(__VA_ARGS__), module_name));   \
    }

#define when_shamlog_error if (shambase::logs::details::loglevel >= LogLevel_Error::logval)

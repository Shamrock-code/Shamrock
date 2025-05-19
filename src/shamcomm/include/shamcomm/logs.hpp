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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shamcmdopt/term_colors.hpp"
#include <string>

/**
 * @brief Namespace containing logs utils
 */
namespace shamcomm::logs {
    /**
     * @namespace details
     * @brief Namespace for internal details of the logs module
     */
    namespace details {
        /**
         * @brief Internal variable to store the global log level
         */
        inline i8 loglevel = 0;
    } // namespace details

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log level manip
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Set the global log level
     *
     * @param val The new log level
     */
    inline void set_loglevel(i8 val) { details::loglevel = val; }

    /**
     * @brief Get the current global log level
     *
     * @return The current log level
     */
    inline i8 get_loglevel() { return details::loglevel; }
} // namespace shamcomm::logs

namespace shamcomm::logs {

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
            return shambase::format("{} ", static_cast<void *>(var1)) + format_message(var2...);
        }

        else {
            // General case for other types
            // Format the argument as a string and concatenate it with the formatted string from the
            // remaining arguments
            return shambase::format("{} ", var1) + format_message(var2...);
        }
    }

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
        shambase::print(shamcomm::logs::format_message(var1, var2...));
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
        shambase::println(shamcomm::logs::format_message(var1, var2...));
        shambase::flush();
    }

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
} // namespace shamcomm::logs

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

}

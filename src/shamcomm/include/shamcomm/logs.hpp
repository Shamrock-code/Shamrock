// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

namespace shamcomm::logs {
    namespace details {
        inline i8 loglevel = 0;
    } // namespace details

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log level manip
    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline void set_loglevel(i8 val) { details::loglevel = val; }
    inline i8 get_loglevel() { return details::loglevel; }
} // namespace shamcomm::logs

namespace shamcomm::logs {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log message formatting
    ////////////////////////////////////////////////////////////////////////////////////////////////
    inline std::string format_message() { return ""; }

    template<typename T, typename... Types>
    std::string format_message(T var1, Types... var2);

    template<typename... Types>
    inline std::string format_message(std::string s, Types... var2) {
        return s + " " + format_message(var2...);
    }

    template<typename T, typename... Types>
    inline std::string format_message(T var1, Types... var2) {
        if constexpr (std::is_same_v<T, const char *>) {
            return std::string(var1) + " " + format_message(var2...);
        } else if constexpr (std::is_pointer_v<T>) {
            return shambase::format("{} ", static_cast<void *>(var1)) + format_message(var2...);
        } else {
            return shambase::format("{} ", var1) + format_message(var2...);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // log message printing
    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline void print() {}

    template<typename T, typename... Types>
    void print(T var1, Types... var2) {
        shambase::print(shamcomm::logs::format_message(var1, var2...));
    }

    inline void print_ln() {}

    template<typename T, typename... Types>
    void print_ln(T var1, Types... var2) {
        shambase::println(shamcomm::logs::format_message(var1, var2...));
        shambase::flush();
    }
} // namespace shamcomm::logs

struct LogLevel_DebugAlloc {
    constexpr static i8 logval              = 127;
    constexpr static const char *level_name = "Debug Alloc";

    static std::string reformat(const std::string &in, std::string module_name);
};

struct LogLevel_DebugMPI {
    constexpr static i8 logval              = 100;
    constexpr static const char *level_name = "Debug MPI";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_DebugSYCL {
    constexpr static i8 logval              = 11;
    constexpr static const char *level_name = "Debug SYCL";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_Debug {
    constexpr static i8 logval              = 10;
    constexpr static const char *level_name = "Debug";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_Info {
    constexpr static i8 logval              = 1;
    constexpr static const char *level_name = "";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_Normal {
    constexpr static i8 logval              = 0;
    constexpr static const char *level_name = "";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_Warning {
    constexpr static i8 logval              = -1;
    constexpr static const char *level_name = "Warning";

    static std::string reformat(const std::string &in, std::string module_name);
};
struct LogLevel_Error {
    constexpr static i8 logval              = -10;
    constexpr static const char *level_name = "Error";

    static std::string reformat(const std::string &in, std::string module_name);
};

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

    template<typename... Types>
    inline void raw(Types... var2) {
        print(var2...);
    }

    template<typename... Types>
    inline void raw_ln(Types... var2) {
        print_ln(var2...);
    }

    inline void print_faint_row() {
        raw_ln(
            shambase::term_colors::faint() + "-----------------------------------------------------"
            + shambase::term_colors::reset());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log levels
    ////////////////////////////////////////////////////////////////////////////////////////////////

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

#define X DECLARE_LOG_LEVEL
    LIST_LEVEL
#undef X

#undef DECLARE_LOG_LEVEL
    ///////////////////////////////////
    // log level declared
    ///////////////////////////////////

#define IsActivePrint(_name, StructREF)                                                            \
    if (details::loglevel >= log_##_name) {                                                        \
        shamcomm::logs::raw("    ");                                                               \
    }                                                                                              \
    _name##_ln("xxx", "xxx", "(", "logger::" #_name, ")");

    inline void print_active_level() {

// logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
#define X IsActivePrint
        LIST_LEVEL
#undef X
        // logger::raw_ln(terminal_effects::faint + "----------------------" +
        // terminal_effects::reset);
    }

#undef IsActivePrint

} // namespace shamcomm::logs

namespace logger {

    using namespace shamcomm::logs;

}

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
#include "shambase/string.hpp"
#include "shamcmdopt/term_colors.hpp"
#include <iostream>
#include <string>

#define LIST_LEVEL                                                                                 \
    X(debug_alloc, shambase::term_colors::col8b_red(), "Debug Alloc ", 127)                        \
    X(debug_mpi, shambase::term_colors::col8b_blue(), "Debug MPI ", 100)                           \
    X(debug_sycl, shambase::term_colors::col8b_magenta(), "Debug SYCL", 11)                        \
    X(debug, shambase::term_colors::col8b_green(), "Debug ", 10)                                   \
    X(info, shambase::term_colors::col8b_cyan(), "", 1)                                            \
    X(normal, shambase::term_colors::empty(), "", 0)                                               \
    X(warn, shambase::term_colors::col8b_yellow(), "Warning ", -1)                                 \
    X(err, shambase::term_colors::col8b_red(), "Error ", -10)

namespace shamcomm::logs {
    namespace details {
        inline i8 loglevel = 0;
    } // namespace details

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log level manip
    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline void set_loglevel(i8 val) { details::loglevel = val; }
    inline i8 get_loglevel() { return details::loglevel; }

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
        std::cout << shamcomm::logs::format_message(var1, var2...);
    }

    inline void print_ln() {}

    template<typename T, typename... Types>
    void print_ln(T var1, Types... var2) {
        std::cout << shamcomm::logs::format_message(var1, var2...) << std::endl;
    }

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
            shambase::term_colors::faint() +
            "-----------------------------------------------------" +
            shambase::term_colors::reset());
    }







    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log levels
    ////////////////////////////////////////////////////////////////////////////////////////////////

    #define DECLARE_LOG_LEVEL(_name, color, loginf, logval)                                                                     \
                                                                                                                            \
    constexpr i8 log_##_name = (logval);                                                                                              \
                                                                                                                            \
    template <typename... Types> inline void _name(std::string module_name, Types... var2) {                                \
        if (details::loglevel >= log_##_name) {                                                                                      \
            std::cout << "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color) + (loginf) +          \
                             shambase::term_colors::reset() + ": ";                                                                \
            shamcomm::logs::print(var2...);                                                                                         \
        }                                                                                                                   \
    }                                                                                                                       \
                                                                                                                            \
    template <typename... Types> inline void _name##_ln(std::string module_name, Types... var2) {                           \
        if (details::loglevel >= log_##_name) {                                                                                      \
            std::cout << "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color) + (loginf) +          \
                             shambase::term_colors::reset() + ": ";                                                                \
            shamcomm::logs::print_ln(var2...);                                                                                         \
        }                                                                                                                     \
    }

    #define X DECLARE_LOG_LEVEL
    LIST_LEVEL
    #undef X

    #undef DECLARE_LOG_LEVEL
    ///////////////////////////////////
    // log level declared
    ///////////////////////////////////



    #define IsActivePrint(_name, color, loginf, logval) \
        if (details::loglevel >= log_##_name) {shamcomm::logs::raw("    ");} _name##_ln("xxx", "xxx","(","logger::" #_name,")");

    inline void print_active_level(){

        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
        #define X IsActivePrint
        LIST_LEVEL
        #undef X
        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);

    }

    #undef IsActivePrint

} // namespace shamcomm::logs


namespace logger {

    using namespace shamcomm::logs;
    
}


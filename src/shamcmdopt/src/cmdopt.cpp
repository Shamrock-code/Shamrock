// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file cmdopt.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/details/generic_opts.hpp"
#include "shamcmdopt/env.hpp"
#include <string_view>
#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Exception handler for exeption in this lib
 */
class ShamCmdOptException : public std::exception {
    public:
    /// Exception CTOR from a message (cstring)
    explicit ShamCmdOptException(const char *message) : msg_(message) {}

    /// Exception CTOR from a message (string)
    explicit ShamCmdOptException(const std::string &message) : msg_(message) {}

    /// Destructor
    virtual ~ShamCmdOptException() noexcept {}

    /// Get the message attached to the exception
    virtual const char *what() const noexcept { return msg_.c_str(); }

    protected:
    /// Held message
    std::string msg_;
};

namespace shamcmdopt {

    /// Error string to be printed in case of failure
    auto err_str = []() {
        return "[" + shambase::term_colors::col8b_red() + "Error" + shambase::term_colors::reset()
               + "]";
    };

    std::string_view executable_name;   ///< Executable name
    std::vector<std::string_view> args; ///< Executable argument list (mapped from argv)
    bool init_done;                     ///< Has cmdopt init been called

    /// Struct for data related to an option
    struct Opts {
        std::string name;                ///< Name of the option (including dashes)
        std::optional<std::string> args; ///< Documention of the option argument
        std::string description;         ///< Description of the otion
    };

    /// Registered cli options
    std::vector<Opts> registered_opts;

    /**
     * @brief Check if the option name is registered
     *
     * @param name the option name (including dashes)
     * @return true the option is registered
     * @return false  the option is not registered
     */
    bool is_name_registered(const std::string_view &name) {
        for (auto opt : registered_opts) {
            if (opt.name == name) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Check if all argument passed to shamrock where registered otherwise throw
     */
    void check_args_registered() {
        bool error = false;

        std::string err_buf;

        for (auto arg : args) {
            if (arg.rfind("-", 0) == 0) {
                if (!is_name_registered(arg)) {
                    fmt::println(
                        err_str() + " opts argument : " + std::string(arg) + " is not registered");
                    err_buf += "\"";
                    err_buf += arg;
                    err_buf += "\"";
                    err_buf += " ";
                    error = true;
                }
            }
        }

        if (error) {
            print_help();
            throw ShamCmdOptException(err_buf + "names are not registered in ::opts");
        }
    }

    /**
     * @brief Check if init has been performed otherwise throw
     */
    void check_init() {
        if (!init_done)
            throw ShamCmdOptException("Cmdopt uninitialized");
    }

    bool has_option(const std::string_view &option_name) {
        check_init(); // We must init the cmdopt before checking if an option is there

        if (!is_name_registered(option_name)) {
            fmt::println(
                err_str() + " opts argument :" + std::string(option_name) + " is not registered");
            throw ShamCmdOptException(
                std::string(option_name) + " option is not registered in ::opts");
        }

        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                return true;
        }

        return false;
    }

    std::string_view get_option(const std::string_view &option_name) {
        check_init();

        if (!is_name_registered(option_name)) {
            fmt::println(
                err_str() + " opts argument :" + std::string(option_name) + "is not registered");
            throw ShamCmdOptException(
                std::string(option_name) + " option is not registered in ::opts");
        }

        for (auto it = args.begin(), end = args.end(); it != end; ++it) {
            if (*it == option_name)
                if (it + 1 != end)
                    return *(it + 1);
        }

        return "";
    }

    void register_opt(std::string name, std::optional<std::string> args, std::string description) {

        for (auto &[n, arg, desc] : registered_opts) {
            if (name == n) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("The option {} is already registered", name));
            }
        }

        registered_opts.push_back({name, args, description});
    }

    /// supplied argc from main
    int argc;

    /// supplied argv from main
    char **argv;

    void init(int _argc, char *_argv[]) {
        argc = _argc;
        argv = _argv;

        register_cmdopt_generic_opts();

        executable_name = std::string_view(argv[0]);
        args            = std::vector<std::string_view>(argv + 1, argv + argc);
        init_done       = true;
        check_args_registered();

        process_cmdopt_generic_opts();
    }

    int get_argc() {
        if (init_done) {
            return argc;
        }
        return 0;
    }
    char **get_argv() {
        if (init_done) {
            return argv;
        }
        return 0;
    }

    void print_help() {
        fmt::println(shambase::format("executable : {}", executable_name));

        fmt::println("\nUsage :");

        std::sort(
            registered_opts.begin(), registered_opts.end(), [](const auto &lhs, const auto &rhs) {
                return lhs.name < rhs.name;
            });

        for (auto &[n, arg, desc] : registered_opts) {

            std::string arg_print = arg.value_or("");

            fmt::println(
                shambase::format_printf(
                    "%-15s %-15s : %s", n.c_str(), arg_print.c_str(), desc.c_str()));
        }
        print_help_env_var();
    }

    bool is_help_mode() {
        if (has_option("--help")) {
            return true;
        } else {
            return false;
        }
    }

} // namespace shamcmdopt

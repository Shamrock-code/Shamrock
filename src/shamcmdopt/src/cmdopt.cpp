// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file cmdopt.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamcmdopt/term_colors.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include <string_view>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

class ShamCmdOptException : public std::exception {
    public:
    explicit ShamCmdOptException(const char *message) : msg_(message) {}

    explicit ShamCmdOptException(const std::string &message) : msg_(message) {}

    virtual ~ShamCmdOptException() noexcept {}

    virtual const char *what() const noexcept { return msg_.c_str(); }

    protected:
    std::string msg_;
};

namespace shamcmdopt {

    auto err_str = []() {
        return "[" + shambase::term_colors::col8b_red() + "Error" + shambase::term_colors::reset()
               + "]";
    };

    std::string_view executable_name;
    std::vector<std::string_view> args;
    bool init_done;

    struct Opts {
        std::string name;
        std::optional<std::string> args;
        std::string description;
    };

    std::vector<Opts> registered_opts;

    bool is_name_registered(const std::string_view &name) {
        for (auto opt : registered_opts) {
            if (opt.name == name) {
                return true;
            }
        }
        return false;
    }

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

    void check_init() {
        if (!init_done)
            throw ShamCmdOptException("Cmdopt uninitialized");
    }

    bool has_option(const std::string_view &option_name) {
        check_init();

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

        registered_opts.push_back({name, args, description});
    }

    int argc;
    char **argv;

    void init(int _argc, char *_argv[]) {
        argc = _argc;
        argv = _argv;

        shamcmdopt::register_opt("--help", {}, "show this message");
        executable_name = std::string_view(argv[0]);
        args            = std::vector<std::string_view>(argv + 1, argv + argc);
        init_done       = true;
        check_args_registered();
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

        for (auto &[n, arg, desc] : registered_opts) {

            std::string arg_print = arg.value_or("");

            fmt::println(shambase::format_printf(
                "%-15s %-15s : %s", n.c_str(), arg_print.c_str(), desc.c_str()));
        }
        print_help_env_var();
    }

    bool is_help_mode() {
        if (has_option("--help")) {
            print_help();
            return true;
        } else {
            return false;
        }
    }

} // namespace shamcmdopt

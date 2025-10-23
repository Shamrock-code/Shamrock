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
 * @file exception_ctx.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include <string>
#include <vector>

namespace shambase {

    /**
     * @brief An argument containing a name and a value
     *
     * This struct is used to store the name and value of an argument when creating
     * exceptions with context.
     */
    struct args_info {
        std::string name;
        std::string value;

        args_info() = default;

        template<class T>
        args_info(std::string name, const T& value) : name(std::move(name)) {
            try {
                this->value = shambase::format("{}", value);
            } catch (const std::exception &e) {
                this->value = "format failed : " + std::string(e.what());
                throw;
            }
        }
    };

    /**
     * @brief A context group containing a section name and a list of arguments
     *
     * This struct is used to group arguments under a named section when creating
     * exceptions with context.
     */
    struct arg_group {
        std::string section_name;
        std::vector<args_info> args;

        arg_group() = default;

        template<typename... Args>
        arg_group(std::string name, Args &&...args_list)
            : section_name(name), args{std::forward<Args>(args_list)...} {}
    };

    /**
     * @brief A context containing a list of argument groups
     *
     * This struct is used to group argument groups under a named section when creating
     * exceptions with context.
     */
    struct context {
        std::vector<arg_group> groups;

        context() = default;

        template<typename... Args>
        context(Args &&...args_list) : groups{std::forward<Args>(args_list)...} {}
    };

    /**
     * @brief Make an exception with a message and variadic context groups
     *
     * This function allows to make an exception with multiple context groups.
     * Each context group has a section name and a list of arguments.
     *
     * Usage:
     * @code{.cpp}
     * throw shambase::make_except_with_loc_with_ctx<std::invalid_argument>(
     *          "The cross product is zero",
     *          {shambase::arg_group{
     *               "function args",
     *               ARG_INFO(center),
     *               ARG_INFO(delta_x),
     *               ARG_INFO(delta_y),
     *               ARG_INFO(nx),
     *               ARG_INFO(ny)},
     *           shambase::arg_group{"internal variables", ARG_INFO(e_z)}});
     * @endcode
     *
     * @tparam ExcptTypes The type of the exception to make
     * @param message The message of the exception
     * @param ctx The context containing the arguments
     * @param loc The source location where the exception is thrown
     * @return ExcptTypes The exception
     */
    template<class ExcptTypes, typename... Contexts>
    inline ExcptTypes make_except_with_loc_with_ctx(
        std::string message, context ctx, SourceLocation loc = SourceLocation{}) {
        std::string msg = message;
        msg += "\nexception context :\n";

        for (const auto &group : ctx.groups) {
            msg += shambase::format("  {}:\n", group.section_name);
            for (const auto &arg : group.args) {
                msg += shambase::format("    {} = {}\n", arg.name, arg.value);
            }
        }

        return shambase::make_except_with_loc<ExcptTypes>(msg, loc);
    }

} // namespace shambase

/**
 * @brief Macro to automatically capture variable name and value for exception context
 *
 * This macro stringifies the variable name and passes both the name and value
 * to create an args_info object.
 *
 * Example:
 * @code{.cpp}
 * int value = 42;
 * SHAM_ARG(value) // Creates args_info("value", 42)
 * @endcode
 */
#define ARG_INFO(var) shambase::args_info(#var, var)

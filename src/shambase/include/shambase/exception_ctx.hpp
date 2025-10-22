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

    struct args_info {
        std::string name;
        std::string value;
        std::optional<std::string> special_print;

        args_info() = default;

        template<class T>
        args_info(std::string name, T value) : name(name), value(shambase::format("{}", value)) {}

        args_info(std::string special_print) : special_print(special_print) {}
    };

    /**
     * @brief Make an exception with a message and a context
     *
     * This function allows to make an exception with a message and a context.
     * The context is a vector of args_info objects, that are used to print the
     * arguments of the exception.
     *
     * Usage :
     * @code{.cpp}
     * throw shambase::make_except_with_loc_with_ctx<std::invalid_argument>(
     *     "The cross product of delta_x and delta_y is zero",
     *     std::vector<shambase::args_info>{
     *         shambase::args_info("--function args--"), // special print
     *         shambase::args_info("center", center), // name and value
     *         shambase::args_info("delta_x", delta_x), // name and value
     *         shambase::args_info("delta_y", delta_y), // name and value
     *         shambase::args_info("nx", nx), // name and value
     *         shambase::args_info("ny", ny), // name and value
     *         shambase::args_info("--internal variables--"), // special print
     *         shambase::args_info("e_z", e_z)}); // name and value
     * @endcode
     *
     * @tparam ExcptTypes The type of the exception to make
     * @param message The message of the exception
     * @param args The arguments of the exception
     * @param loc The location of the exception
     * @return ExcptTypes The exception
     */
    template<class ExcptTypes>
    inline ExcptTypes make_except_with_loc_with_ctx(
        std::string message, std::vector<args_info> args, SourceLocation loc = SourceLocation{}) {
        std::string msg = message;
        msg += "\n  context :\n";
        for (const auto &arg : args) {
            if (arg.special_print) {
                msg += shambase::format("    {}\n", arg.special_print.value());
            } else {
                msg += shambase::format("    {} = {}\n", arg.name, arg.value);
            }
        }
        return shambase::make_except_with_loc<ExcptTypes>(msg, loc);
    }
} // namespace shambase

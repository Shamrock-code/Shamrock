// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file env.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "fmt/core.h"
#include "shamcmdopt/env.hpp"
#include <optional>
#include <utility>
#include <vector>

std::optional<std::string> shamcmdopt::getenv_str(const char *env_var) {
    const char *val = std::getenv(env_var);
    if (val != nullptr) {
        return std::string(val);
    }
    return {};
}

/// List of documented env variables
std::vector<std::pair<std::string, std::string>> env_var_reg = {};

void shamcmdopt::register_env_var_doc(std::string env_var, std::string desc) {

    for (auto &[_env_var, _desc] : env_var_reg) {
        if (_env_var == env_var) {
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("The env var {} is already registered", env_var));
        }
    }

    env_var_reg.push_back({env_var, desc});
}

void shamcmdopt::print_help_env_var() {

    if (env_var_reg.empty()) {
        return;
    }

    auto stringify = [](std::optional<std::string> val) -> std::string {
        if (val) {
            return shambase::format("= {}", *val);
        }
        return "";
    };

    fmt::println("\nEnv variables :");
    for (const auto &[var, desc] : env_var_reg) {
        auto val = getenv_str(var.c_str());

        fmt::println(shambase::format("  {:<29} : {}", var, desc));

        if (val) {
            fmt::println(shambase::format("    = {}", *val));
        }
    }
}

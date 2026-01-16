// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file locate_pylib.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/print.hpp"
#include "shambindings/pybindings.hpp"
#include "shamcmdopt/env.hpp"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// env var to set the path to the pylib
std::optional<std::string> pylib_path_env_var = shamcmdopt::getenv_str("SHAMROCK_PYLIB_PATH");

/// @brief Path to shamrock utils lib supplied at configure time
extern std::vector<std::string> configure_time_pylib_paths();

namespace shambindings {

    std::optional<std::string> get_binary_path() {

        // first try /proc/self/exe
        try {
            return std::filesystem::read_symlink("/proc/self/exe");
        } catch (const std::filesystem::filesystem_error &e) {
            return std::nullopt;
        }

        // then try sys.executable from python because why not XD
        try {
            py::module_ sys        = py::module_::import("sys");
            std::string executable = sys.attr("executable").cast<std::string>();
            return executable;
        } catch (const std::exception &e) {
            return std::nullopt;
        }
    }

    std::optional<std::string> is_path_valid_pylib(const std::string &path) {
        std::filesystem::path path_fs(path);
        if (!std::filesystem::exists(path_fs)) {
            return "does not exist";
        }
        if (!std::filesystem::is_directory(path_fs)) {
            return "is not a directory";
        }

        // it should contain shamrock folder
        if (!std::filesystem::exists(path_fs / "shamrock")) {
            return "does not contain shamrock folder";
        }

        // it should be a directory
        if (!std::filesystem::is_directory(path_fs / "shamrock")) {
            return "shamrock folder is not a directory";
        }

        // it should contain shamrock/__init__.py
        if (!std::filesystem::exists(path_fs / "shamrock" / "__init__.py")) {
            return "shamrock/__init__.py does not exist";
        }

        return std::nullopt;
    }

    std::string locate_pylib_path(bool do_print) {

        auto get_binary_dir = []() -> std::filesystem::path {
            auto bpath = get_binary_path();
            if (bpath.has_value()) {
                return std::filesystem::path(bpath.value()).parent_path();
            }
            return std::filesystem::path(".");
        };

        // Get the path to the current binary
        std::filesystem::path binary_dir = get_binary_dir();

        std::filesystem::path pyshamrock_path_relative1 = binary_dir / ".." / "pylib";
        std::filesystem::path pyshamrock_path_relative2 = binary_dir / ".." / "src" / "pylib";

        std::vector<std::string> possible_paths
            = {"pyshamrock", pyshamrock_path_relative1, pyshamrock_path_relative2};

        for (auto path : configure_time_pylib_paths()) {
            possible_paths.push_back(path);
        }

        if (pylib_path_env_var.has_value()) {
            shambase::println("using pylib path from env var: " + pylib_path_env_var.value());
            possible_paths = {pylib_path_env_var.value()};
        }

        std::optional<std::string> ret = std::nullopt;

        for (const auto &path : possible_paths) {
            auto err = is_path_valid_pylib(path);
            if (err.has_value()) {
                shambase::println("pylib path " + path + " is not valid: " + err.value());
            } else {
                ret = path;
                break;
            }
        }

        if (ret.has_value()) {
            if (do_print) {
                shambase::println("using pylib path : " + ret.value());
            }
            return ret.value();
        }

        shambase::println("pylib path was not found ... Something might be broken");

        return "";
    };

} // namespace shambindings

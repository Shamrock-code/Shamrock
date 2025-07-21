// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file start_python.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/popen.hpp"
#include "shambase/print.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pybindings.hpp"
#include "shambindings/start_python.hpp"
#include <pybind11/embed.h>
#include <cstdlib>
#include <optional>
#include <string>

/**
 * @brief path of the script to generate sys.path
 *
 * @return const char*
 */
extern const char *configure_time_py_sys_path();

/// @brief path of the python executable that was used to configure sys.path
extern const char *configure_time_py_executable();

/**
 * @brief Script to run ipython
 *
 */
extern const char *run_ipython_src();

/// value use to set the value of sys.path if set by the user at runtime
std::optional<std::string> runtime_set_pypath = std::nullopt;

/**
 * @brief Retrieves the Python path to be used for the application.
 *
 * This function returns the Python path that should be used, prioritizing
 * the runtime-set value if available. If no runtime value is set, it
 * defaults to the path configured during the application's build time.
 *
 * @return std::string The Python path to be used.
 */
std::string get_pypath() {

    if (runtime_set_pypath.has_value()) {
        return runtime_set_pypath.value();
    }
    return configure_time_py_sys_path();
}

/// Script to check that the python distrib is the expected one
std::string check_python_is_excpeted_version = R"(

import sys
cur_path = os.path.realpath(current_path)

# This is broken on MacOS and give shamrock instead i don't know why ... stupid python ...
#sysyexec = os.path.realpath(sys.executable)
# So the fix is to check that the resolved path starts with base_exec_prefix
# see https://docs.python.org/3/library/sys.html#sys.base_prefix
sysprefix = os.path.realpath(sys.base_exec_prefix)

#if cur_path != sysyexec:
if not cur_path.startswith(sysprefix):
    print("Current python is not the expected version, you may be using mismatched Pythons.")
    print("Current path : ",cur_path)
    #print("Expected path : ",sysyexec)
    print("Expected prefix : ",sysprefix)

)";

namespace shambindings {

    void setpypath(std::string path) { runtime_set_pypath = path; }

    void setpypath_from_binary(std::string binary_path) {

        std::string cmd    = binary_path + " -c \"import sys;print(sys.path, end= '')\"";
        runtime_set_pypath = shambase::popen_fetch_output(cmd.c_str());
    }

    void modify_py_sys_path(bool do_print) {

        if (do_print) {
            shambase::println(
                "Shamrock configured with Python path : \n    "
                + std::string(configure_time_py_executable()));
        }

        std::string check_py
            = std::string("current_path = \"") + configure_time_py_executable() + "\"\n";
        check_py += check_python_is_excpeted_version;
        py::exec(check_py);

        std::string modify_path = std::string("paths = ") + get_pypath() + "\n";
        modify_path += R"(import sys;sys.path = paths)";
        py::exec(modify_path);
    }

    void start_ipython(bool do_print) {

        py::scoped_interpreter guard{};
        modify_py_sys_path(do_print);

        if (do_print) {
            shambase::println("--------------------------------------------");
            shambase::println("-------------- ipython ---------------------");
            shambase::println("--------------------------------------------");
        }
        py::exec(run_ipython_src());
        if (do_print) {
            shambase::println("--------------------------------------------");
            shambase::println("------------ ipython end -------------------");
            shambase::println("--------------------------------------------\n");
        }
    }

    void run_py_file(std::string file_path, bool do_print) {
        py::scoped_interpreter guard{};
        modify_py_sys_path(do_print);

        if (do_print) {
            shambase::println("-----------------------------------");
            shambase::println("running pyscript : " + file_path);
            shambase::println("-----------------------------------");
        }
        py::eval_file(file_path);
        if (do_print) {
            shambase::println("-----------------------------------");
            shambase::println("pyscript end");
            shambase::println("-----------------------------------");
        }
    }
} // namespace shambindings

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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/popen.hpp"
#include <cstdlib>
#include <optional>
#if defined(DOXYGEN) || defined(SHAMROCK_EXECUTABLE_BUILD)

    #include "shambase/print.hpp"
    #include "shambindings/pybindaliases.hpp"
    #include "shambindings/start_python.hpp"
    #include <string>

/**
 * @brief path of the script to generate sys.path
 *
 * @return const char*
 */
extern const char *configure_time_py_sys_path();

/// @brief path of the python executable that was used to configure sys.path
const char *configure_time_py_executable();

/**
 * @brief Script to run ipython
 *
 */
extern const char *run_ipython_src();

std::optional<std::string> runtime_set_pypath = std::nullopt;

std::string get_pypath() {
    if (runtime_set_pypath.has_value()) {
        return runtime_set_pypath.value();
    }
    return configure_time_py_sys_path();
}

std::string check_python_is_excpeted_version = R"(

import sys
cur_path = os.path.realpath(current_path)

# This is broken on MacOS and give shamrock instead i don't know why ... stupid python ...
#sysyexec = os.path.realpath(sys.executable)
sysprefix = os.path.realpath(sys.prefix)

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

    void modify_py_sys_path() {

        shambase::println(
            "Shamrock configured with Python path : \n    "
            + std::string(configure_time_py_executable()));

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
        modify_py_sys_path();

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
        modify_py_sys_path();

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

#endif

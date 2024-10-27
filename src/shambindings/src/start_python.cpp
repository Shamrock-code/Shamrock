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
extern const char *change_py_sys_path();

/**
 * @brief Script to run ipython
 *
 */
extern const char *run_ipython_src();

/**
 * @brief Python script to modify sys.path to point to the correct libraries
 *
 */
const std::string modify_path = std::string("paths = ") + change_py_sys_path() + "\n" +
                                R"(
import sys
sys.path = paths
)";

namespace shambindings {

    void start_ipython(bool do_print) {
        py::scoped_interpreter guard{};
        py::exec(modify_path);

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
        py::exec(modify_path);

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

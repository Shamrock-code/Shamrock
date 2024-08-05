// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
extern const char *ipython_run_src();

/**
 * @brief Script to run ipython
 * 
 */
const std::string run_ipython_src =
    R"(

from IPython import start_ipython
from traitlets.config.loader import Config

import signal

# here the signal interup for sigint is None
# this make ipython freaks out for weird reasons
# registering the handler fix it ...
# i swear python c api is horrible to works with
import shamrock.sys
signal.signal(signal.SIGINT, shamrock.sys.signal_handler)

c = Config()

banner ="SHAMROCK Ipython terminal\n" + "Python %s\n"%sys.version.split("\n")[0]

c.TerminalInteractiveShell.banner1 = banner

c.TerminalInteractiveShell.banner2 = """### 
import shamrock
###
"""

start_ipython(config=c)

)";

/**
 * @brief Python script to modify sys.path to point to the correct libraries
 * 
 */
const std::string modify_path = std::string("paths = ") + ipython_run_src() + "\n" +
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
        py::exec(run_ipython_src);
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
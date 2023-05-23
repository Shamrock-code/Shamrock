// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "aliases.hpp"
#include "shambase/stacktrace.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamsys/SignalCatch.hpp"
#include "shambase/time.hpp"
#include "shamtest/shamtest.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "shambindings/pybindaliases.hpp"

//%Impl status : Should rewrite

const std::string run_ipython_src = R"(
from IPython import start_ipython
from traitlets.config.loader import Config
import sys

c = Config()

banner ="SHAMROCK Ipython terminal\n" + "Python %s\n"%sys.version.split("\n")[0]

c.TerminalInteractiveShell.banner1 = banner

c.TerminalInteractiveShell.banner2 = """### 
import shamrock
###
"""

start_ipython(config=c)

)";


int main(int argc, char *argv[]) {
    
    {

    
    StackEntry stack_loc{};




    std::cout << shamrock_title_bar_big << std::endl;

    opts::register_opt("--sycl-cfg","(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--loglevel","(logvalue)", "specify a log level");

    opts::register_opt("--nocolor",{}, "disable colored ouput");

    opts::register_opt("--rscript","(filepath)", "run shamrock with python runscirpt");
    opts::register_opt("--ipython",{}, "run shamrock in Ipython mode");

    opts::init(argc, argv);

    if(opts::is_help_mode()){
        return 0;
    }

    if(opts::has_option("--nocolor")){
        terminal_effects::disable_colors();
    }



    if(opts::has_option("--loglevel")){
        std::string level = std::string(opts::get_option("--loglevel"));

        i32 a = atoi(level.c_str());

        if(i8(a) != a){
            logger::err_ln("Cmd OPT", "you must select a loglevel in a 8bit integer range");
        }

        logger::loglevel = a;

        if(a == i8_max){
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln("-> modified loglevel to",logger::loglevel,"enabled log types : ");
        logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
        logger::print_active_level();
        logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
    }




    if(opts::has_option("--sycl-cfg")){
        shamsys::instance::init(argc,argv);
    }

    





    shamsys::register_signals();
    //*
    {
        namespace py = pybind11;
        //RunScriptHandler rscript;
        
        if(opts::has_option("--ipython")){
            StackEntry stack_loc{};

            py::scoped_interpreter guard{};
            
            
            std::cout << "--------------------------------------------" << std::endl;
            std::cout << "-------------- ipython ---------------------" << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
            py::exec(run_ipython_src);
            std::cout << "--------------------------------------------" << std::endl;
            std::cout << "------------ ipython end -------------------" << std::endl;
            std::cout << "--------------------------------------------\n" << std::endl;

            //rscript.run_ipython();
        }else if(opts::has_option("--rscript")){
            StackEntry stack_loc{};
            std::string fname = std::string(opts::get_option("--rscript"));
            //RunScriptHandler rscript;
            //rscript.run_file(fname);

            py::scoped_interpreter guard{};

            std::cout << "-----------------------------------" << std::endl;
            std::cout << "running pyscript : " << fname << std::endl;
            std::cout << "-----------------------------------" << std::endl;
            py::eval_file(fname);
            std::cout << "-----------------------------------" << std::endl;
            std::cout << "pyscript end" << std::endl;
            std::cout << "-----------------------------------" << std::endl;


        }else{



            //SimulationSPH<TestTimestepper, TestSimInfo>::run_sim();
            //SimulationSPH<TestTimestepperSync, TestSimInfo>::run_sim();
        }
        
    }
    //*/





    





    }

    shambase::details::dump_profiling(shamsys::instance::world_rank);

    shamsys::instance::close();

    
}
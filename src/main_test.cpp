// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//%Impl status : Good

#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtest/shamtest.hpp"

int main(int argc, char *argv[]){


    opts::register_opt("--sycl-cfg","(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--sycl-ls",{}, "list available devices");
    opts::register_opt("--sycl-ls-map",{}, "list available devices & list of queue bindings");
    opts::register_opt("--loglevel","(logvalue)", "specify a log level");
    opts::register_opt("--nocolor",{}, "disable colored ouput");
    opts::register_opt("--benchmark-mpi",{}, "micro benchmark for MPI");

    opts::register_opt("--test-list",{}, "print test availables");
    opts::register_opt("--run-only",{"(test name)"}, "run only this test");
    opts::register_opt("--full-output",{}, "print the assertions in the tests");

    opts::register_opt("--benchmark",{}, "run only benchmarks");
    opts::register_opt("--analysis",{}, "run only analysis");
    opts::register_opt("--unittest",{}, "run only unittest");


    opts::register_opt("-o",{"(filepath)"}, "output test report in that file");


    opts::init(argc, argv);
    if(opts::is_help_mode()){
        return 0;
    }


    if(opts::has_option("--loglevel")){
        std::string level = std::string(opts::get_option("--loglevel"));

        i32 a = atoi(level.c_str());

        if(i8(a) != a){
            logger::err_ln("Cmd OPT", "you must select a loglevel in a 8bit integer range");
        }

        logger::loglevel = a;

    }

    if(opts::has_option("--sycl-cfg")){
        shamsys::instance::init(argc,argv);
    }

    if(shamsys::instance::world_rank == 0){
        std::cout << shamrock_title_bar_big << std::endl;
        logger::print_faint_row();

        std::cout <<"\n"<< terminal_effects::colors_foreground_8b::cyan + "Git infos "+ terminal_effects::reset+":\n";
        std::cout << git_info_str <<std::endl;

        logger::print_faint_row();

        logger::raw_ln("MPI status : ");

        logger::raw_ln(" - MPI & SYCL init :",terminal_effects::colors_foreground_8b::green + "Ok"+ terminal_effects::reset);

        shamsys::instance::print_mpi_capabilities();

        shamsys::instance::check_dgpu_available();
        
    }

    shamsys::instance::validate_comm();

    if(opts::has_option("--benchmark-mpi")){
        shamsys::run_micro_benchmark();
    }

    if(shamsys::instance::world_rank == 0){
        logger::print_faint_row();
        logger::raw_ln("log status : ");
        if(logger::loglevel == i8_max){
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln(" - Loglevel :",u32(logger::loglevel),", enabled log types : ");
        logger::print_active_level();
    
    } 

    

    if(opts::has_option("--sycl-ls")){

        if(shamsys::instance::world_rank == 0){
            logger::print_faint_row();
        }
        shamsys::instance::print_device_list();
        
    }

    if(opts::has_option("--sycl-ls-map")){

        if(shamsys::instance::world_rank == 0){
            logger::print_faint_row();
        }
        shamsys::instance::print_device_list();
        shamsys::instance::print_queue_map();
        
    }

    

    


    if(shamsys::instance::world_rank == 0){
        logger::print_faint_row();
        logger::raw_ln(" - Code init",terminal_effects::colors_foreground_8b::green + "DONE"+ terminal_effects::reset, "now it's time to",
        terminal_effects::colors_foreground_8b::cyan + terminal_effects::blink + "ROCK"+ terminal_effects::reset);
        logger::print_faint_row();
    }
    

    bool run_bench    = opts::has_option("--benchmark") ;
    bool run_analysis = opts::has_option("--analysis") ;
    bool run_unittest = opts::has_option("--unittest") ;

    if((run_bench || run_unittest || run_analysis) == false){
        run_bench = true;
        run_analysis = true;
        run_unittest = true;
    }
    
    return shamtest::run_all_tests(argc,argv,run_bench,run_analysis,run_unittest);
    
}
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file main_test.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shamcomm/worldInfo.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "version.hpp"

int main(int argc, char *argv[]) {

    opts::register_opt("--sycl-cfg", "(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--sycl-ls", {}, "list available devices");
    opts::register_opt("--sycl-ls-map", {}, "list available devices & list of queue bindings");
    opts::register_opt("--loglevel", "(logvalue)", "specify a log level");
    opts::register_opt("--nocolor", {}, "disable colored ouput");
    opts::register_opt("--benchmark-mpi", {}, "micro benchmark for MPI");

    opts::register_opt("--test-list", {}, "print test availables");
    opts::register_opt("--run-only", {"(test name)"}, "run only this test");
    opts::register_opt("--full-output", {}, "print the assertions in the tests");

    opts::register_opt("--benchmark", {}, "run only benchmarks");
    opts::register_opt("--validation", {}, "run only validation tests");
    opts::register_opt("--unittest", {}, "run only unittest");
    opts::register_opt("--long-test", {}, "run also long tests");

    opts::register_opt("-o", {"(filepath)"}, "output test report in that file");

    opts::init(argc, argv);
    if (opts::is_help_mode()) {
        return 0;
    }

    if (opts::has_option("--loglevel")) {
        std::string level = std::string(opts::get_option("--loglevel"));

        i32 a = atoi(level.c_str());

        if (i8(a) != a) {
            logger::err_ln("Cmd OPT", "you must select a loglevel in a 8bit integer range");
        }

        logger::loglevel = a;
    }

    if (opts::has_option("--sycl-cfg")) {
        shamsys::instance::init(argc, argv);
    }

    if (shamcomm::world_rank() == 0) {
        std::cout << shamrock_title_bar_big << std::endl;
        logger::print_faint_row();

        std::cout << "\n"
                  << terminal_effects::colors_foreground_8b::cyan + "Git infos " +
                         terminal_effects::reset + ":\n";
        std::cout << git_info_str << std::endl;

        logger::print_faint_row();

        logger::raw_ln("MPI status : ");

        logger::raw_ln(
            " - MPI & SYCL init :",
            terminal_effects::colors_foreground_8b::green + "Ok" + terminal_effects::reset);

        shamsys::instance::print_mpi_capabilities();

        shamsys::instance::check_dgpu_available();
    }

    shamsys::instance::validate_comm();

    if (opts::has_option("--benchmark-mpi")) {
        shamsys::run_micro_benchmark();
    }

    if (shamcomm::world_rank() == 0) {
        logger::print_faint_row();
        logger::raw_ln("log status : ");
        if (logger::loglevel == i8_max) {
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln(" - Loglevel :", u32(logger::loglevel), ", enabled log types : ");
        logger::print_active_level();
    }

    if (opts::has_option("--sycl-ls")) {

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
        }
        shamsys::instance::print_device_list();
    }

    if (opts::has_option("--sycl-ls-map")) {

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
        }
        shamsys::instance::print_device_list();
        shamsys::instance::print_queue_map();
    }

    if (shamcomm::world_rank() == 0) {
        logger::print_faint_row();
        logger::raw_ln(
            " - Code init",
            terminal_effects::colors_foreground_8b::green + "DONE" + terminal_effects::reset,
            "now it's time to",
            terminal_effects::colors_foreground_8b::cyan + terminal_effects::blink + "ROCK" +
                terminal_effects::reset);
        logger::print_faint_row();
    }

    shamtest::TestConfig cfg{};

    cfg.print_test_list_exit = false;

    cfg.full_output = false;

    cfg.output_tex = true;
    if (opts::has_option("-o")) {
        if (opts::get_option("-o").size() == 0) {
            opts::print_help();
        }
        cfg.json_output = opts::get_option("-o");
    }

    cfg.run_long_tests = opts::has_option("--long-test");

    cfg.run_benchmark  = opts::has_option("--benchmark");
    cfg.run_validation = opts::has_option("--validation");
    cfg.run_unittest   = opts::has_option("--unittest");
    if ((cfg.run_benchmark || cfg.run_unittest || cfg.run_validation) == false) {
        cfg.run_unittest   = true;
        cfg.run_validation = true;
        cfg.run_benchmark  = false;
    }

    if (opts::has_option("--run-only")) {
        cfg.run_only = opts::get_option("--run-only");
    }

    return shamtest::run_all_tests(argc, argv, cfg);
}
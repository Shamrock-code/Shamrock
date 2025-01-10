// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file main_test.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/fpe_except.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "version.hpp"

int main(int argc, char *argv[]) {

    opts::register_opt("--sycl-cfg", "(idcomp:idalt) ", "specify the compute & alt queue index");
    opts::register_opt("--sycl-ls", {}, "list available devices");
    opts::register_opt("--sycl-ls-map", {}, "list available devices & list of queue bindings");
    opts::register_opt("--loglevel", "(logvalue)", "specify a log level");
    opts::register_opt("--benchmark-mpi", {}, "micro benchmark for MPI");

    opts::register_opt("--test-list", {}, "print test availables");
    opts::register_opt("--gen-test-list", {}, "print test availables");
    opts::register_opt("--run-only", {"(test name)"}, "run only this test");
    opts::register_opt("--full-output", {}, "print the assertions in the tests");

    opts::register_opt("--benchmark", {}, "run only benchmarks");
    opts::register_opt("--validation", {}, "run only validation tests");
    opts::register_opt("--unittest", {}, "run only unittest");
    opts::register_opt("--long-test", {}, "run also long tests");
    opts::register_opt("--force-dgpu-on", {}, "for direct mpi comm on");
    opts::register_opt("--force-dgpu-off", {}, "for direct mpi comm off");

    opts::register_opt("-o", {"(filepath)"}, "output test report in that file");

    shamcmdopt::register_opt("--feenableexcept", "", "Enable FPE exceptions");

    opts::register_env_var_doc("REF_FILES_PATH", "reference test files path");
    shamcmdopt::register_env_var_doc("SHAMLOGFORMATTER", "Change the log formatter (values :0-3)");

    shamcmdopt::register_env_var_doc("SHAM_PROF_PREFIX", "Prefix of shamrock profile outputs");
    shamcmdopt::register_env_var_doc("SHAM_PROF_USE_NVTX", "Enable NVTX profiling");
    shamcmdopt::register_env_var_doc("SHAM_PROFILING", "Enable Shamrock profiling");
    shamcmdopt::register_env_var_doc(
        "SHAM_PROF_USE_COMPLETE_EVENT",
        "Use complete event instead of begin end for chrome tracing");
    shamcmdopt::register_env_var_doc(
        "SHAM_PROF_EVENT_RECORD_THRES", "Change the event recording threshold");
    shamcmdopt::register_env_var_doc("SHAMLOG_ERR_ON_EXCEPT", "Enable logging of exceptions");

    opts::init(argc, argv);
    if (opts::is_help_mode()) {
        return 0;
    }

    if (opts::has_option("--feenableexcept")) {
        sham::enable_fpe_exceptions();
    }

    if (opts::has_option("--gen-test-list")) {
        std::string_view outfile = opts::get_option("--gen-test-list");
        shamtest::gen_test_list(outfile);
        return 0;
    }

    if (opts::has_option("--loglevel")) {
        std::string level = std::string(opts::get_option("--loglevel"));

        i32 a = atoi(level.c_str());

        if (i8(a) != a) {
            logger::err_ln("Cmd OPT", "you must select a loglevel in a 8bit integer range");
        }

        logger::set_loglevel(a);
    }

    if (opts::has_option("--sycl-cfg")) {
        shamsys::instance::init(argc, argv);
    }

    if (shamcomm::world_rank() == 0) {
        print_title_bar();

        logger::print_faint_row();

        logger::raw_ln("MPI status : ");

        logger::raw_ln(
            " - MPI & SYCL init :",
            shambase::term_colors::col8b_green() + "Ok" + shambase::term_colors::reset());

        shamsys::instance::print_mpi_capabilities();

        shamsys::instance::check_dgpu_available();
    }

    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
    shamcomm::validate_comm(sptr);

    if (opts::has_option("--benchmark-mpi")) {
        shamsys::run_micro_benchmark();
    }

    if (shamcomm::world_rank() == 0) {
        logger::print_faint_row();
        logger::raw_ln("log status : ");
        if (logger::get_loglevel() == i8_max) {
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln(" - Loglevel :", u32(logger::get_loglevel()), ", enabled log types : ");
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
            shambase::term_colors::col8b_green() + "DONE" + shambase::term_colors::reset(),
            "now it's time to",
            shambase::term_colors::col8b_cyan() + shambase::term_colors::blink() + "ROCK"
                + shambase::term_colors::reset());
        logger::print_faint_row();
    }

    shamtest::TestConfig cfg{};

    cfg.print_test_list_exit = false;

    cfg.full_output = opts::has_option("--full-output");

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

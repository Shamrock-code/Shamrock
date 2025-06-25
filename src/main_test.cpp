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

#include "shambase/string.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/fpe_except.hpp"
#include "shambindings/pybindings.hpp"
#include "shambindings/start_python.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/version.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/shamrock_smi.hpp"
#include "shamtest/shamtest.hpp"
#include <pybind11/embed.h>

/// Call bindings init for the shamrock python module
PYBIND11_EMBEDDED_MODULE(shamrock, m) { shambindings::init_embed(m); }

int main(int argc, char *argv[]) {

    opts::register_opt("--sycl-cfg", "(idcomp:idalt) ", "specify the compute & alt queue index");

    opts::register_opt(
        "--smi", {}, "print information about all available SYCL devices in the cluster");
    opts::register_opt(
        "--smi-full", {}, "print information about EVERY available SYCL devices in the cluster");

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

    opts::register_opt("--pypath", "(sys.path)", "python sys.path to set");
    opts::register_opt("--pypath-from-bin", "(python binary)", "set sys.path from python binary");

    opts::register_opt("-o", {"(filepath)"}, "output test report in that file");

    shamcmdopt::register_opt("--feenableexcept", "", "Enable FPE exceptions");

    opts::register_env_var_doc("REF_FILES_PATH", "reference test files path");

    shamcmdopt::register_env_var_doc("SHAM_PROF_PREFIX", "Prefix of shamrock profile outputs");
    shamcmdopt::register_env_var_doc("SHAM_PROF_USE_NVTX", "Enable NVTX profiling");
    shamcmdopt::register_env_var_doc("SHAM_PROFILING", "Enable Shamrock profiling");
    shamcmdopt::register_env_var_doc(
        "SHAM_PROF_USE_COMPLETE_EVENT",
        "Use complete event instead of begin end for chrome tracing");
    shamcmdopt::register_env_var_doc(
        "SHAM_PROF_EVENT_RECORD_THRES", "Change the event recording threshold");

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
            shambase::throw_with_loc<std::invalid_argument>(
                "you must select a loglevel in a 8bit integer range");
        } else {
            logger::set_loglevel(i8(a));
        }
    }

    if (opts::has_option("--sycl-cfg")) {
        shamsys::instance::init(argc, argv);
    } else {
        using namespace shamsys::instance;
        start_mpi(MPIInitInfo{opts::get_argc(), opts::get_argv()});
        if (shamcomm::world_rank() == 0) {
            logger::warn_ln(
                "Init", "No kernel can be run without a sycl configuration (--sycl-cfg x:x)");
        }
    }

    if (shamcomm::world_rank() == 0) {
        print_title_bar();

        logger::print_faint_row();
        if (shamsys::instance::is_initialized()) {
            logger::raw_ln("MPI status : ");

            shamsys::instance::print_mpi_comm_info();

            logger::raw_ln(
                " - MPI & SYCL init     :",
                shambase::term_colors::col8b_green() + "Ok" + shambase::term_colors::reset());

            shamsys::instance::print_mpi_capabilities();
        }
    }

    if (shamsys::instance::is_initialized()) {
        shamsys::instance::check_dgpu_available();
        auto sptr = shamsys::instance::get_compute_scheduler_ptr();
        shamcomm::validate_comm(sptr);
    }

    if (shamcomm::world_rank() == 0) {
        logger::print_faint_row();
        logger::print_active_level();
    }

    bool is_smi      = opts::has_option("--smi");
    bool is_smi_full = opts::has_option("--smi-full");

    if (is_smi || is_smi_full) {
        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
        }
        shamsys::shamrock_smi(is_smi_full);
    }

    if (opts::has_option("--benchmark-mpi")) {
        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
        }
        if (shamsys::instance::is_initialized()) {
            shamsys::run_micro_benchmark();
        } else {
            logger::warn_ln(
                "Init",
                "--benchmark-mpi can't be run without a sycl configuration (--sycl-cfg x:x)");
        }
    }

    if (shamsys::instance::is_initialized()) {
        bool _ = shamrock::are_experimental_features_allowed();
        shamcomm::logs::code_init_done_log();

        if (opts::has_option("--pypath")) {
            shambindings::setpypath(std::string(opts::get_option("--pypath")));
        }

        if (opts::has_option("--pypath-from-bin")) {
            std::string pybin = std::string(opts::get_option("--pypath-from-bin"));
            shambindings::setpypath_from_binary(pybin);
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

    } else {

        if (shamcomm::world_rank() == 0) {
            logger::warn_ln(
                "Init", "No sycl configuration (--sycl-cfg x:x) has been set, early exit");
        }
        shamsys::instance::close_mpi();
        return 0;
    }
}

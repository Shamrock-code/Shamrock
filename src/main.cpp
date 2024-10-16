// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file main.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * @version 0.1
 * @date 2022-05-24
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/start_python.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SignalCatch.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "version.hpp"
#include <type_traits>
#include <unordered_map>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

//%Impl status : Should rewrite

int main(int argc, char *argv[]) {

    {

        StackEntry stack_loc{};

        opts::register_opt("--sycl-ls", {}, "list available devices");
        opts::register_opt("--sycl-ls-map", {}, "list available devices & list of queue bindings");
        opts::register_opt("--benchmark-mpi", {}, "micro benchmark for MPI");

        opts::register_opt(
            "--sycl-cfg", "(idcomp:idalt) ", "specify the compute & alt queue index");
        opts::register_opt("--loglevel", "(logvalue)", "specify a log level");

        opts::register_opt("--rscript", "(filepath)", "run shamrock with python runscirpt");
        opts::register_opt("--ipython", {}, "run shamrock in Ipython mode");
        opts::register_opt("--force-dgpu-on", {}, "for direct mpi comm on");
        opts::register_opt("--force-dgpu-off", {}, "for direct mpi comm off");

        shamcmdopt::register_env_var_doc(
            "SHAMLOGFORMATTER", "Change the log formatter (values :0-3)");

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
                logger::raw_ln(
                    "If you've seen spam in your life i can garantee you, this is worst");
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

        shamsys::register_signals();
        {

            if (opts::has_option("--ipython")) {
                StackEntry stack_loc{};

                if (shamcomm::world_size() > 1) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "cannot run ipython mode with > 1 processes");
                }

                shambindings::start_ipython(true);

            } else if (opts::has_option("--rscript")) {
                StackEntry stack_loc{};
                std::string fname = std::string(opts::get_option("--rscript"));

                shambindings::run_py_file(fname, shamcomm::world_rank() == 0);

            } else {
                logger::raw_ln("Nothing to do ... exiting");
            }
        }
    }

#ifdef SHAMROCK_USE_PROFILING
// shambase::details::dump_profiling(shamcomm::world_rank());
#endif

    shamsys::instance::close();
}

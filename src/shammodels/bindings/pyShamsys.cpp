// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamsys.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambindings/pyNodeInstance.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamsys/legacy/log.hpp"
#include "version.hpp"
#include "shamsys/SignalCatch.hpp"

Register_pymod(pysyslibinit) {

    m.def(
        "change_loglevel",
        [](u32 loglevel) {
            if (loglevel > i8_max) {
                throw shambase::make_except_with_loc<std::invalid_argument>("loglevel must be below 128");
            }

            if (loglevel == i8_max) {
                logger::raw_ln(
                    "If you've seen spam in your life i can garantee you, this is worst");
            }

            logger::raw_ln("-> modified loglevel to", logger::get_loglevel(), "enabled log types : ");

            logger::set_loglevel(loglevel);
            logger::print_active_level();
        },
        R"pbdoc(

        Change the loglevel

    )pbdoc");

    m.def(
        "get_git_info",
        []() {
            return git_info_str;
        },
        R"pbdoc(
        Return git_info_str
    )pbdoc");

    m.def(
        "print_git_info",
        []() {
            logger::raw_ln(git_info_str);
        },
        R"pbdoc(
        print git_info_str
    )pbdoc");

    m.def(
        "get_compile_arg",
        []() {
            return compile_arg;
        },
        R"pbdoc(
        Return compile_arg
    )pbdoc");

    m.def(
        "print_compile_arg",
        []() {
            logger::raw_ln(compile_arg);
        },
        R"pbdoc(
        print compile_arg
    )pbdoc");

    m.def(
        "dump_profiling",
        [](std::string prefix) {
            shambase::details::dump_profilings(prefix, shamcomm::world_rank());
        },
        R"pbdoc(
        dump profiling data
    )pbdoc");

    m.def(
        "dump_profiling_chrome",
        [](std::string prefix) {
            shambase::details::dump_profilings_chrome(prefix, shamcomm::world_rank());
        },
        R"pbdoc(
        dump profiling data
    )pbdoc");

    

    m.def(
        "clear_profiling_data",
        []() {
            shambase::details::clear_profiling_data();
        },
        R"pbdoc(
        dump profiling data
    )pbdoc");


    py::module sys_module = m.def_submodule("sys", "system handling part of shamrock");
    sys_module.def("signal_handler",&shamsys::details::signal_callback_handler);


    shamsys::instance::register_pymodules(sys_module);
}

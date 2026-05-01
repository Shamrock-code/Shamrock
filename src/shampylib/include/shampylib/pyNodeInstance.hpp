// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pyNodeInstance.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcomm/mpiInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/stl.h>

namespace shamsys::instance {
    void register_pymodules(py::module &m) {

        using namespace shamsys::instance;

        m.def(
            "init",
            [](std::string sycl_cfg) {
                init_sycl_mpi(sycl_cfg, MPIInitInfo{opts::get_argc(), opts::get_argv()});
            },
            R"pbdoc(

            The init function for shamrock node instance

            )pbdoc");

        m.def(
            "close",
            []() {
                close();
            },
            R"pbdoc(

            The close function for shamrock node instance

            )pbdoc");

        m.def("get_process_name", &shamcomm::get_process_name, R"pbdoc(

            Get the name of the process

            )pbdoc");

        m.def(
            "world_rank",
            []() {
                return shamcomm::world_rank();
            },
            R"pbdoc(
            Get the world rank
            )pbdoc");

        m.def(
            "world_size",
            []() {
                return shamcomm::world_size();
            },
            R"pbdoc(
            Get the world size
            )pbdoc");

        m.def(
            "is_initialized",
            []() {
                return is_initialized();
            },
            R"pbdoc(
            Return true if the node instance is initialized
            )pbdoc");

        m.def(
            "mpi_barrier",
            []() {
                shamcomm::mpi::Barrier(MPI_COMM_WORLD);
            },
            R"pbdoc(
            Call the MPI barrier
            )pbdoc");

        m.def(
            "get_compute_device_properties",
            []() {
                auto &sched = shamsys::instance::get_compute_scheduler();
                auto &dev   = shambase::get_check_ref(sched.ctx).device;
                auto &prop  = shambase::get_check_ref(dev).prop;

                py::dict dict;
                dict["vendor"]                     = sham::vendor_name(prop.vendor);
                dict["backend"]                    = sham::backend_name(prop.backend);
                dict["type"]                       = sham::device_type_name(prop.type);
                dict["name"]                       = prop.name;
                dict["platform"]                   = prop.platform;
                dict["global_mem_size"]            = prop.global_mem_size;
                dict["global_mem_cache_line_size"] = prop.global_mem_cache_line_size;
                dict["global_mem_cache_size"]      = prop.global_mem_cache_size;
                dict["local_mem_size"]             = prop.local_mem_size;
                dict["max_compute_units"]          = prop.max_compute_units;
                dict["max_mem_alloc_size_dev"]     = prop.max_mem_alloc_size_dev;
                dict["max_mem_alloc_size_host"]    = prop.max_mem_alloc_size_host;
                dict["mem_base_addr_align"]        = prop.mem_base_addr_align;
                // dict["sub_group_sizes"]            = prop.sub_group_sizes;
                // dict["default_work_group_size"]    = prop.default_work_group_size;
                // dict["pci_address"]                = prop.pci_address;
                // dict["warnings"]                   = prop.warnings;

                return dict;
            },
            R"pbdoc(
        Get the properties of the compute device
    )pbdoc");

        m.def(
            "get_microbench_results",
            []() {
                return shamsys::get_microbench_results();
            },
            R"pbdoc(
            Get the microbench results
            )pbdoc");
    }
} // namespace shamsys::instance

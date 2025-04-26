// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file shamrock_smi.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/for_each_device.hpp"
#include <unordered_map>
#include <functional>
#include <string>
#include <vector>

namespace shamsys {

    void shamrock_smi_summary() {
        if (!shamcomm::is_mpi_initialized()) {
            shambase::throw_with_loc<std::runtime_error>("MPI should be initialized");
        }

        u32 rank = shamcomm::world_rank();

        auto add_array = [&](std::string &print, std::string header, std::string data) {
            print
                += (header
                    + "----------------------------------------------------------------------------"
                      "---------------")
                       .substr(0, 91)
                   + "\n";
            print += ("| id |      Device name          |      Platform name     |  Type  | "
                      "   Memsize   | units |\n");
            print += ("----------------------------------------------------------------------------"
                      "---------------\n");
            print += (data);
            print += ("----------------------------------------------------------------------------"
                      "---------------\n");
        };

        std::string nodeconfig = "";
        for_each_device([&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName   = dev.get_info<sycl::info::device::name>();

            auto device = sham::sycl_dev_to_sham_dev(key_global, dev);

            std::string devname  = shambase::trunc_str(DeviceName, 25);
            std::string platname = shambase::trunc_str(PlatformName, 22);
            std::string devtype  = shambase::trunc_str(sham::device_type_name(device.prop.type), 6);

            f64 mem            = device.prop.global_mem_size;
            std::string memstr = shambase::readable_sizeof(mem);

            nodeconfig += shambase::format(
                              "| {:>2} | {:>25.25} | {:>22.22} | {:>6} | {:>12} | {:>5} | ",
                              key_global,
                              devname,
                              platname,
                              devtype,
                              memstr,
                              device.prop.max_compute_units)
                          + "\n";
        });

        std::unordered_map<std::string, int> nodeconfig_histogram
            = shamcomm::string_histogram({nodeconfig}, "@#%");

        if (rank == 0) {
            std::string print = "Available devices :\n";
            for (auto &[node_conf, count] : nodeconfig_histogram) {
                std::string arr = "";
                add_array(arr, shambase::format("{} x Nodes: ", count), node_conf);
                print += shambase::format("\n{}\n", arr);
            }
            printf("%s", print.data());
        }
    }

    /// Print SMI for all devices
    void shamrock_smi_all() {
        if (!shamcomm::is_mpi_initialized()) {
            shambase::throw_with_loc<std::runtime_error>("MPI should be initialized");
        }

        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        for_each_device([&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName   = dev.get_info<sycl::info::device::name>();

            auto device = sham::sycl_dev_to_sham_dev(key_global, dev);

            std::string devname  = shambase::trunc_str(DeviceName, 25);
            std::string platname = shambase::trunc_str(PlatformName, 22);
            std::string devtype  = shambase::trunc_str(sham::device_type_name(device.prop.type), 6);

            f64 mem            = device.prop.global_mem_size;
            std::string memstr = shambase::readable_sizeof(mem);

            print_buf += shambase::format(
                             "| {:>4} | {:>2} | {:>25.25} | {:>22.22} | {:>6} | {:>12} | {:>5} | ",
                             rank,
                             key_global,
                             devname,
                             platname,
                             devtype,
                             memstr,
                             device.prop.max_compute_units)
                         + "\n";
        });

        std::string recv;
        shamcomm::gather_str(print_buf, recv);
        if (rank == 0) {
            std::string print = "Available devices :\n";
            print += ("----------------------------------------------------------------------------"
                      "----------------------\n");
            print += ("| rank | id |      Device name          |      Platform name     |  Type  | "
                      "   Memsize   | units |\n");
            print += ("----------------------------------------------------------------------------"
                      "----------------------\n");
            print += (recv);
            print += ("----------------------------------------------------------------------------"
                      "----------------------\n");
            printf("%s\n", print.data());
        }
    }

    /// Print SMI for selected devices
    void shamrock_smi_selected(bool list_all_devices) {
        if (!shamcomm::is_mpi_initialized()) {
            shambase::throw_with_loc<std::runtime_error>("MPI should be initialized");
        }

        if (shamsys::instance::is_initialized()) {

            if (list_all_devices) {
                shamsys::instance::print_queue_map();
            }

            sham::Device &dev
                = shambase::get_check_ref(instance::get_compute_scheduler().ctx->device);

            std::string DeviceName = dev.dev.get_info<sycl::info::device::name>();

            std::string dev_with_id = shambase::format("{} (id={})", DeviceName, dev.device_id);

            std::unordered_map<std::string, int> devicename_histogram
                = shamcomm::string_histogram({dev_with_id}, "\n");

            f64 mem           = dev.prop.global_mem_size;
            u64 compute_units = dev.prop.max_compute_units;

            f64 total_mem       = shamalgs::collective::allreduce_sum(mem);
            u64 n_compute_units = shamalgs::collective::allreduce_sum(compute_units);

            if (shamcomm::world_rank() == 0) {

                std::string print = "Selected devices : (totals can we wrong if using multiple "
                                    "rank per devices)\n";

                for (auto &[key, value] : devicename_histogram) {
                    print += shambase::format("  - {} x {}", value, key) + "\n";
                }
                print += "  Total memory : " + shambase::readable_sizeof(total_mem) + "\n";
                print += "  Total compute units : " + std::to_string(n_compute_units) + "\n";

                printf("%s\n", print.data());
            }
        }
    }

    void shamrock_smi(bool list_all_devices) {
        if (shamcomm::world_rank() == 0) {
            logger::raw_ln(" ----- Shamrock SMI ----- \n");
        }

        if (list_all_devices) {
            shamrock_smi_all();
        } else {
            shamrock_smi_summary();
        }

        shamrock_smi_selected(list_all_devices);
    }

} // namespace shamsys

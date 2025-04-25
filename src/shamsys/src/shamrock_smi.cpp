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
#include <functional>

std::string SHAMROCK_LIST_ALL_DEVICES = shamcmdopt::getenv_str_default_register(
    "SHAMROCK_LIST_ALL_DEVICES", "0", "List all available devices");

namespace shamsys {

    void shamrock_smi() {
        if (!shamcomm::is_mpi_initialized()) {
            shambase::throw_with_loc<std::runtime_error>("MPI should be initialized");
        }

        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        u32 ncpu            = 0;
        u32 ngpu            = 0;
        u32 naccelerator    = 0;
        u64 n_compute_units = 0;
        f64 total_mem       = 0;

        for_each_device([&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName   = dev.get_info<sycl::info::device::name>();

            auto device = sham::sycl_dev_to_sham_dev(key_global, dev);

            std::string devname  = shambase::trunc_str(DeviceName, 25);
            std::string platname = shambase::trunc_str(PlatformName, 22);
            std::string devtype  = shambase::trunc_str(sham::device_type_name(device.prop.type), 6);

            switch (device.prop.type) {
            case sham::DeviceType::CPU: ncpu++; break;
            case sham::DeviceType::GPU: ngpu++; break;
            default: naccelerator++;
            }

            f64 mem = device.prop.global_mem_size;
            total_mem += mem;
            std::string memstr = shambase::readable_sizeof(mem);

            n_compute_units += device.prop.max_compute_units;

            print_buf += shambase::format(
                             "| {:>4} | {:>2} | {:>25.25} | {:>22.22} | {:>6} | {:>12} | {:>8} | ",
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

        u32 ncpu_global            = shamalgs::collective::allreduce_sum(ncpu);
        u32 ngpu_global            = shamalgs::collective::allreduce_sum(ngpu);
        u32 naccelerator_global    = shamalgs::collective::allreduce_sum(naccelerator);
        f64 total_mem_global       = shamalgs::collective::allreduce_sum(total_mem);
        u64 n_compute_units_global = shamalgs::collective::allreduce_sum(n_compute_units);

        if (rank == 0) {
            std::string print = "Cluster SYCL Info : \n";
            print += ("----------------------------------------------------------------------------"
                      "-------------------------\n");
            print += ("| rank | id |      Device name          |      Platform name     |  Type  | "
                      "   Memsize   | c. units |\n");
            print += ("----------------------------------------------------------------------------"
                      "-------------------------\n");
            print += (recv);
            print += ("----------------------------------------------------------------------------"
                      "-------------------------\n");
            printf("%s\n", print.data());

            printf("Summary : \n");
            printf("  Number of ranks : %u\n", shamcomm::world_size());
            printf("  Number of CPU devices : %u\n", ncpu_global);
            printf("  Number of GPU devices : %u\n", ngpu_global);
            printf("  Number of accelerator devices : %u\n", naccelerator_global);
            printf("  Number of compute units : %llu\n", n_compute_units_global);
            printf("  Total memory : %s\n\n", shambase::readable_sizeof(total_mem_global).data());
            shamcomm::logs::print_faint_row();
        }
    }

} // namespace shamsys

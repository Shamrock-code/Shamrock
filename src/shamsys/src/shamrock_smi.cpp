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

#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/for_each_device.hpp"
#include <functional>

namespace shamsys {

    void shamrock_smi() {
        if (!shamcomm::is_mpi_initialized()) {
            shambase::throw_with_loc<std::runtime_error>("MPI should be initialized");
        }

        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        for_each_device([&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName   = dev.get_info<sycl::info::device::name>();

            std::string devname  = shambase::trunc_str(DeviceName, 29);
            std::string platname = shambase::trunc_str(PlatformName, 24);
            std::string devtype  = shambase::trunc_str(shambase::getDevice_type(dev), 6);

            print_buf += shambase::format(
                             "| {:>4} | {:>2} | {:>29.29} | {:>24.24} | {:>6} |",
                             rank,
                             key_global,
                             devname,
                             platname,
                             devtype)
                         + "\n";
        });

        std::string recv;
        shamcomm::gather_str(print_buf, recv);

        if (rank == 0) {
            std::string print = "Cluster SYCL Info : \n";
            print += ("----------------------------------------------------------------------------"
                      "----\n");
            print += ("| rank | id |        Device name            |       Platform name      |  "
                      "Type  |\n");
            print += ("----------------------------------------------------------------------------"
                      "----\n");
            print += (recv);
            print += ("----------------------------------------------------------------------------"
                      "----");
            printf("%s\n", print.data());
        }
    }

} // namespace shamsys

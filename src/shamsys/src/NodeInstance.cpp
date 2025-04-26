// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeInstance.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/tty.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiInfo.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/EnvVariables.hpp"
#include "shamsys/MpiDataTypeHandler.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/change_log_format.hpp"
#include "shamsys/device_select.hpp"
#include "shamsys/for_each_device.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace shamsys::instance::details {

    void print_device_list() {
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

} // namespace shamsys::instance::details

namespace syclinit {

    bool initialized = false;

    std::shared_ptr<sham::Device> device_compute;
    std::shared_ptr<sham::Device> device_alt;

    std::shared_ptr<sham::DeviceContext> ctx_compute;
    std::shared_ptr<sham::DeviceContext> ctx_alt;

    std::shared_ptr<sham::DeviceScheduler> sched_compute;
    std::shared_ptr<sham::DeviceScheduler> sched_alt;

    void init_device_scheduling() {
        StackEntry stack_loc{false};
        ctx_compute = std::make_shared<sham::DeviceContext>(device_compute);
        ctx_alt     = std::make_shared<sham::DeviceContext>(device_alt);

        sched_compute = std::make_shared<sham::DeviceScheduler>(ctx_compute);
        sched_alt     = std::make_shared<sham::DeviceScheduler>(ctx_alt);

        sched_compute->test();
        sched_alt->test();

        // logger::raw_ln("--- Compute ---");
        // sched_compute->print_info();
        // logger::raw_ln("--- Alternative ---");
        // sched_alt->print_info();
    }

    void init_queues(std::string search_key) {
        StackEntry stack_loc{false};

        auto devs = shamsys::select_devices(search_key);

        device_alt     = std::move(devs.device_alt);
        device_compute = std::move(devs.device_compute);

        init_device_scheduling();
        initialized = true;
    }

    void finalize() {
        initialized = false;

        device_compute.reset();
        device_alt.reset();

        ctx_compute.reset();
        ctx_alt.reset();

        sched_compute.reset();
        sched_alt.reset();
    }
}; // namespace syclinit

namespace shamsys::instance {

    u32 compute_queue_eu_count = 64;

    u32 get_compute_queue_eu_count(u32 id) { return compute_queue_eu_count; }

    bool is_initialized() { return syclinit::initialized && shamcomm::is_mpi_initialized(); };

    void print_queue_map() {
        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        std::optional<u32> loc = env::get_local_rank();
        if (loc) {
            print_buf = shambase::format(
                "| {:>4} | {:>8} | {:>12} | {:>16} |\n",
                rank,
                *loc,
                syclinit::device_alt->device_id,
                syclinit::device_compute->device_id);
        } else {
            print_buf = shambase::format(
                "| {:>4} | {:>8} | {:>12} | {:>16} |\n",
                rank,
                "???",
                syclinit::device_alt->device_id,
                syclinit::device_compute->device_id);
        }

        std::string recv;
        shamcomm::gather_str(print_buf, recv);

        if (rank == 0) {
            std::string print = "Queue map : \n";
            print += ("----------------------------------------------------\n");
            print += ("| rank | local id | alt queue id | compute queue id |\n");
            print += ("----------------------------------------------------\n");
            print += (recv);
            print += ("----------------------------------------------------");
            printf("%s\n\n", print.data());
        }
    }

    namespace tmp {

        void print_device_list_debug() {
            u32 rank = 0;

            std::string print_buf = "device avail : \n";

            for_each_device(
                [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
                    auto PlatformName = plat.get_info<sycl::info::platform::name>();
                    auto DeviceName   = dev.get_info<sycl::info::device::name>();

                    std::string devname  = DeviceName;
                    std::string platname = PlatformName;
                    std::string devtype  = "truc";

                    print_buf += std::to_string(key_global) + " " + devname + " " + platname + "\n";
                });

            logger::debug_sycl_ln("InitSYCL", print_buf);
        }

    } // namespace tmp

    void start_sycl_auto(std::string search_key) {
        // start sycl

        tmp::print_device_list_debug();

        if (syclinit::initialized) {
            throw ShamsysInstanceException("Sycl is already initialized");
        }

        if (shamcomm::world_rank() == 0) {
            logger::debug_ln("Sys", "start sycl queues ...");
        }

        syclinit::init_queues(search_key);
    }

    void start_mpi(MPIInitInfo mpi_info) {

        shamcomm::fetch_mpi_capabilities(mpi_info.forced_state);

        mpi::init(&mpi_info.argc, &mpi_info.argv);

        shamcomm::fetch_world_info();

        // now that MPI is started we can use the formatter with rank info
        shamsys::change_log_format();

        if (shamcomm::world_size() < 1) {
            throw ShamsysInstanceException("world size is < 1");
        }

        if (shamcomm::world_rank() < 0) {
            throw ShamsysInstanceException("world size is above i32_max");
        }

        int error;
        // error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

        if (error != MPI_SUCCESS) {
            throw ShamsysInstanceException("failed setting the MPI error mode");
        }

        logger::debug_ln(
            "Sys",
            shambase::format(
                "[{:03}]: \x1B[32mMPI_Init : node n°{:03} | world size : {} | name = {}\033[0m",
                shamcomm::world_rank(),
                shamcomm::world_rank(),
                shamcomm::world_size(),
                shamcomm::get_process_name()));

        mpi::barrier(MPI_COMM_WORLD);
        // if(world_rank == 0){
        if (shamcomm::world_rank() == 0) {
            logger::debug_ln("NodeInstance", "------------ MPI init ok ------------");
            logger::debug_ln("NodeInstance", "creating MPI type for interop");
        }
        create_sycl_mpi_types();
        if (shamcomm::world_rank() == 0) {
            logger::debug_ln("NodeInstance", "MPI type for interop created");
            logger::debug_ln("NodeInstance", "------------ MPI / SYCL init ok ------------");
        }
        mpidtypehandler::init_mpidtype();
    }

    auto init_strategy = shamcmdopt::getenv_str_default_register(
        "SHAM_MPI_INIT_STRATEGY",
        "syclfirst",
        "Select the MPI init strategy (mpifirst, syclfirst) [default: syclfirst]");

    void init_sycl_mpi(std::string search_key, MPIInitInfo mpi_info) {

        if (init_strategy == "syclfirst") {
            start_sycl_auto(search_key);
            start_mpi(mpi_info);
        } else if (init_strategy == "mpifirst") {
            start_mpi(mpi_info);
            start_sycl_auto(search_key);
        } else {
            shambase::throw_unimplemented();
        }

        syclinit::device_compute->update_mpi_prop();
        syclinit::device_alt->update_mpi_prop();
    }

    void init(int argc, char *argv[]) {

        std::optional<shamcomm::StateMPI_Aware> forced_state = std::nullopt;

        if (shamcmdopt::has_option("--force-dgpu-on")) {
            forced_state = shamcomm::StateMPI_Aware::ForcedYes;
        }

        if (shamcmdopt::has_option("--force-dgpu-off")) {
            forced_state = shamcomm::StateMPI_Aware::ForcedNo;
        }

        if (opts::has_option("--sycl-cfg")) {

            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            // logger::debug_ln("NodeInstance", "chosen sycl config :",sycl_cfg);

            init_sycl_mpi(sycl_cfg, {argc, argv, forced_state});

        } else {

            logger::err_ln("NodeInstance", "Please specify a sycl configuration (--sycl-cfg x:x)");
            // std::cout << "[NodeInstance] Please specify a sycl configuration (--sycl-cfg x:x)" <<
            // std::endl;
            throw ShamsysInstanceException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }
    }

    void close_mpi() {
        mpidtypehandler::free_mpidtype();

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
            logger::raw_ln(" - MPI finalize \nExiting ...\n");
            logger::raw_ln(" Hopefully it was quick :')\n");
        }

        mpi::finalize();
    }

    void close() {

        close_mpi();

        syclinit::finalize();
    }

    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    sycl::queue &get_compute_queue(u32 /*id*/) { return syclinit::sched_compute->get_queue().q; }

    sycl::queue &get_alt_queue(u32 /*id*/) { return syclinit::sched_alt->get_queue().q; }

    sham::DeviceScheduler &get_compute_scheduler() { return *syclinit::sched_compute; }

    sham::DeviceScheduler &get_alt_scheduler() { return *syclinit::sched_alt; }

    std::shared_ptr<sham::DeviceScheduler> get_compute_scheduler_ptr() {
        return syclinit::sched_compute;
    }

    std::shared_ptr<sham::DeviceScheduler> get_alt_scheduler_ptr() { return syclinit::sched_alt; }

    void print_device_info(const sycl::device &Device) {
        std::cout << "   - " << Device.get_info<sycl::info::device::name>() << " "
                  << shambase::readable_sizeof(
                         Device.get_info<sycl::info::device::global_mem_size>())
                  << "\n";
    }

    void print_device_list() { details::print_device_list(); }

    ////////////////////////////
    // MPI related routines
    ////////////////////////////

    void print_mpi_capabilities() { shamcomm::print_mpi_capabilities(); }

    void check_dgpu_available() {

        using namespace shambase::term_colors;

        u32 loc_use_direct_gpu
            = shambase::get_check_ref(syclinit::device_compute).mpi_prop.is_mpi_direct_capable;

        u32 num_dgpu_use = shamalgs::collective::allreduce_sum(loc_use_direct_gpu);

        if (shamcomm::world_rank() == 0) {
            if (num_dgpu_use == shamcomm::world_size()) {
                logger::raw_ln(shambase::format(
                    " - MPI use Direct Comm : {}", col8b_green() + "Yes" + reset()));
            } else if (num_dgpu_use > 0) {
                logger::raw_ln(shambase::format(
                    " - MPI use Direct Comm : {} ({} of {})",
                    col8b_yellow() + "Partial" + reset(),
                    num_dgpu_use,
                    shamcomm::world_size()));
            } else {
                logger::raw_ln(
                    shambase::format(" - MPI use Direct Comm : {}", col8b_red() + "No" + reset()));
            }
        }
    }

} // namespace shamsys::instance

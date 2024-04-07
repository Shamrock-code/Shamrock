// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeInstance.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "NodeInstance.hpp"

#include "shambackends/Device.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiInfo.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/EnvVariables.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"

#include "MpiDataTypeHandler.hpp"
#include <optional>
#include <stdexcept>
#include <string>

namespace shamsys::instance::details {

    /**
     * @brief for each SYCL device
     *
     * @param fct
     * @return u32 the number of devices
     */
    u32
    for_each_device(std::function<void(u32, const sycl::platform &, const sycl::device &)> fct) {

        u32 key_global        = 0;
        const auto &Platforms = sycl::platform::get_platforms();
        for (const auto &Platform : Platforms) {
            const auto &Devices = Platform.get_devices();
            for (const auto &Device : Devices) {
                fct(key_global, Platform, Device);
                key_global++;
            }
        }
        return key_global;
    }

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
                             devtype) +
                         "\n";
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

    std::unique_ptr<sham::DeviceScheduler> sched_compute;
    std::unique_ptr<sham::DeviceScheduler> sched_alt;

    void init_device_scheduling(){
        StackEntry stack_loc{false};
        ctx_compute = std::make_shared<sham::DeviceContext>(device_compute);
        ctx_alt = std::make_shared<sham::DeviceContext>(device_alt);

        sched_compute = std::make_unique<sham::DeviceScheduler>(ctx_compute);
        sched_alt = std::make_unique<sham::DeviceScheduler>(ctx_alt);

        sched_compute->test();
        sched_alt->test();

        //logger::raw_ln("--- Compute ---");
        //sched_compute->print_info();
        //logger::raw_ln("--- Alternative ---");
        //sched_alt->print_info();
    }

    void init_queues_auto(std::string search_key) {
        StackEntry stack_loc{false};
        std::optional<u32> local_id = shamsys::env::get_local_rank();

        if (local_id) {

            u32 valid_dev_cnt = 0;

            shamsys::instance::details::for_each_device(
                [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
                    if (shambase::contain_substr(
                            plat.get_info<sycl::info::platform::name>(), search_key)) {
                        valid_dev_cnt++;
                    }
                });

            u32 valid_dev_id = 0;

            shamsys::instance::details::for_each_device(
                [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
                    if (shambase::contain_substr(
                            plat.get_info<sycl::info::platform::name>(), search_key)) {

                        if ((*local_id) % valid_dev_cnt == valid_dev_id) {
                            logger::debug_sycl_ln(
                                "Sys",
                                "create queue :\n",
                                "Local ID :",
                                *local_id,
                                "\n Queue id :",
                                key_global);

                            auto PlatformName = plat.get_info<sycl::info::platform::name>();
                            auto DeviceName   = dev.get_info<sycl::info::device::name>();
                            logger::debug_sycl_ln(
                                "NodeInstance",
                                "init alt queue  : ",
                                "|",
                                DeviceName,
                                "|",
                                PlatformName,
                                "|",
                                shambase::getDevice_type(dev),
                                "|");

                            device_alt = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));

                            logger::debug_sycl_ln(
                                "NodeInstance",
                                "init comp queue : ",
                                "|",
                                DeviceName,
                                "|",
                                PlatformName,
                                "|",
                                shambase::getDevice_type(dev),
                                "|");
                            device_compute = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));

                        }

                        valid_dev_id++;
                    }
                });

        } else {
            logger::err_ln("Sys", "cannot query local rank cannot use autodetect");
            throw shambase::make_except_with_loc<std::runtime_error>(
                "cannot query local rank cannot use autodetect");
        }

        init_device_scheduling();
        initialized = true;
    }

    void init_queues(u32 alt_id, u32 compute_id) {
        StackEntry stack_loc{false};

        u32 cnt_dev = shamsys::instance::details::for_each_device(
            [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {});

        if (alt_id >= cnt_dev) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the alt queue id is larger than the number of queue");
        }

        if (compute_id >= cnt_dev) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the compute queue id is larger than the number of queue");
        }

        shamsys::instance::details::for_each_device(
            [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
                auto PlatformName = plat.get_info<sycl::info::platform::name>();
                auto DeviceName   = dev.get_info<sycl::info::device::name>();

                if (key_global == alt_id) {
                    logger::debug_sycl_ln(
                        "NodeInstance",
                        "init alt queue  : ",
                        "|",
                        DeviceName,
                        "|",
                        PlatformName,
                        "|",
                        shambase::getDevice_type(dev),
                        "|");
                        device_alt = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));

                }

                if (key_global == compute_id) {
                    logger::debug_sycl_ln(
                        "NodeInstance",
                        "init comp queue : ",
                        "|",
                        DeviceName,
                        "|",
                        PlatformName,
                        "|",
                        shambase::getDevice_type(dev),
                        "|");
                        device_compute = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));

                }
            });

        init_device_scheduling();
        initialized = true;
    }

    void finalize(){
        initialized = false;

        device_compute.reset();
        device_alt.reset();

        ctx_compute.reset();
        ctx_alt.reset();

        sched_compute.reset();
        sched_alt.reset();   
    }
};

namespace shamsys::instance {

    u32 compute_queue_eu_count = 64;

    u32 get_compute_queue_eu_count(u32 id) { return compute_queue_eu_count; }

    bool is_initialized() {
        int flag = false;
        mpi::initialized(&flag);
        return syclinit::initialized && flag;
    };

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
            printf("%s\n", print.data());
        }
    }


    namespace tmp{



        void print_device_list_debug() {
            u32 rank = 0;

            std::string print_buf = "device avail : ";

            details::for_each_device([&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
                auto PlatformName = plat.get_info<sycl::info::platform::name>();
                auto DeviceName   = dev.get_info<sycl::info::device::name>();

                std::string devname  = DeviceName;
                std::string platname = PlatformName;
                std::string devtype  = "truc";

                print_buf += 
                                std::to_string(key_global) + " " +
                                devname+
                                platname +
                            "\n";
            });

        
            logger::debug_sycl_ln("InitSYCL", print_buf);
        }


    }

    void init(int argc, char *argv[]) {

        tmp::print_device_list_debug();



        if (opts::has_option("--sycl-cfg")) {

            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            // logger::debug_ln("NodeInstance", "chosen sycl config :",sycl_cfg);

            bool force_aware = opts::has_option("--force-dgpu");

            if (shambase::contain_substr(sycl_cfg, "auto:")) {

                std::string search = sycl_cfg.substr(5);
                init_auto(search, MPIInitInfo{argc, argv,force_aware});

            } else {

                size_t split_alt_comp = 0;
                split_alt_comp        = sycl_cfg.find(":");

                if (split_alt_comp == std::string::npos) {
                    logger::err_ln("NodeInstance", "sycl-cfg layout should be x:x");
                    throw ShamsysInstanceException("sycl-cfg layout should be x:x");
                }

                std::string alt_cfg  = sycl_cfg.substr(0, split_alt_comp);
                std::string comp_cfg = sycl_cfg.substr(split_alt_comp + 1, sycl_cfg.length());

                u32 ialt, icomp;
                try {
                    try {
                        ialt = std::stoi(alt_cfg);
                    } catch (const std::invalid_argument &a) {
                        logger::err_ln("NodeInstance", "alt config is not an int");
                        throw ShamsysInstanceException("alt config is not an int");
                    }
                } catch (const std::out_of_range &a) {
                    logger::err_ln("NodeInstance", "alt config is to big for an integer");
                    throw ShamsysInstanceException("alt config is to big for an integer");
                }

                try {
                    try {
                        icomp = std::stoi(comp_cfg);
                    } catch (const std::invalid_argument &a) {
                        logger::err_ln("NodeInstance", "compute config is not an int");
                        throw ShamsysInstanceException("compute config is not an int");
                    }
                } catch (const std::out_of_range &a) {
                    logger::err_ln("NodeInstance", "compute config is to big for an integer");
                    throw ShamsysInstanceException("compute config is to big for an integer");
                }

                init(SyclInitInfo{ialt, icomp}, MPIInitInfo{argc, argv, force_aware});
            }

        } else {

            logger::err_ln("NodeInstance", "Please specify a sycl configuration (--sycl-cfg x:x)");
            // std::cout << "[NodeInstance] Please specify a sycl configuration (--sycl-cfg x:x)" <<
            // std::endl;
            throw ShamsysInstanceException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }
    }

    void start_mpi(MPIInitInfo mpi_info) {

#ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_DEFINE:MPI_COMM_WORLD=" << MPI_COMM_WORLD << "\n";
#endif

        shamcomm::fetch_mpi_capabilities(mpi_info.force_aware);

        mpi::init(&mpi_info.argc, &mpi_info.argv);

        shamcomm::fetch_world_info();

#ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_VALUE:world_size=" << world_size << "\n";
        std::cout << "%MPI_VALUE:world_rank=" << world_rank << "\n";
#endif

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

        syclinit::device_compute->update_mpi_prop();
        syclinit::device_alt->update_mpi_prop();
    }

    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info) {

        start_sycl(sycl_info.alt_queue_id, sycl_info.compute_queue_id);

        start_mpi(mpi_info);
    }

    void init_auto(std::string search_key, MPIInitInfo mpi_info) {

        start_sycl_auto(search_key);

        start_mpi(mpi_info);
    }

    void close() {

        mpidtypehandler::free_mpidtype();

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
            logger::raw_ln(" - MPI finalize \nExiting ...\n");
            logger::raw_ln(" Hopefully it was quick :')\n");
        }
        mpi::finalize();
        syclinit::finalize();
    }

    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    sycl::queue &get_compute_queue(u32 /*id*/) { 
        return syclinit::sched_compute->get_queue().q; 
        }

    sycl::queue &get_alt_queue(u32 /*id*/) { 
        return syclinit::sched_alt->get_queue().q; }

    sham::DeviceScheduler & get_compute_scheduler(){
        return *syclinit::sched_compute;
    }    
    
    sham::DeviceScheduler & get_alt_scheduler(){
        return *syclinit::sched_alt;
    }

    void print_device_info(const sycl::device &Device) {
        std::cout << "   - " << Device.get_info<sycl::info::device::name>() << " "
                  << shambase::readable_sizeof(
                         Device.get_info<sycl::info::device::global_mem_size>())
                  << "\n";
    }

    void print_device_list() { details::print_device_list(); }


    void start_sycl(u32 alt_id, u32 compute_id) {
        // start sycl

        if (syclinit::initialized) {
            throw ShamsysInstanceException("Sycl is already initialized");
        }

        if (shamcomm::world_rank() == 0) {
            logger::debug_ln("Sys", "start sycl queues ...");
        }

        syclinit::init_queues(alt_id, compute_id);
    }

    void start_sycl_auto(std::string search_key) {
        // start sycl

        if (syclinit::initialized) {
            throw ShamsysInstanceException("Sycl is already initialized");
        }

        syclinit::init_queues_auto(search_key);
    }

    ////////////////////////////
    // MPI related routines
    ////////////////////////////

    void print_mpi_capabilities() { shamcomm::print_mpi_capabilities(); }

    void check_dgpu_available() {

        using namespace shambase::term_colors;
        if (syclinit::device_compute->mpi_prop.is_mpi_direct_capable) {
            logger::raw_ln(" - MPI use Direct Comm :", col8b_green() + "Yes" + reset());
        } else {
            logger::raw_ln(" - MPI use Direct Comm :", col8b_red() + "No" + reset());
        }
    }

} // namespace shamsys::instance
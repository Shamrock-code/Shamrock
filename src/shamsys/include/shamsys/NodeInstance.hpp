// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeInstance.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file describing a Node Instance
 *
 * A NodeInstance is a wrapper to perform a combined initialisation of both SYCL & MPI.
 * Essentially it handle the MPI process and corresponding accelerators
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shamcomm/mpiInfo.hpp"
#include "shamcomm/worldInfo.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace shamsys::instance {

    /**
     * @brief Exception type for the NodeInstance
     */
    class ShamsysInstanceException : public std::exception {
        public:
        explicit ShamsysInstanceException(const char *message) : msg_(message) {}

        explicit ShamsysInstanceException(const std::string &message) : msg_(message) {}

        ~ShamsysInstanceException() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

        protected:
        std::string msg_;
    };

    /**
     * @brief Struct containing Sycl Init informations
     * Usage
     * ```
     * SyclInitInfo{alt_id, compute_id}
     * ```
     */
    struct SyclInitInfo {
        u32 alt_queue_id;
        u32 compute_queue_id;
    };

    /**
     * @brief Struct containing MPI Init informations
     * Usage
     * ```
     * MPIInitInfo{argc, argv}
     * ```
     */
    struct MPIInitInfo {
        int argc;
        char **argv;
        std::optional<shamcomm::StateMPI_Aware> forced_state;
    };

    /**
     * @brief to check whether the NodeInstance is initialized
     * @return true NodeInstance is initialized
     * @return false  NodeInstance is not initialized
     */
    bool is_initialized();

    /**
     * @brief initialize the NodeInstance from command line args in the main
     * ```
     * int main(int argc, char *argv[]){
     *     shamsys::instance::init(argc,argv);
     *     ... do stuff ...
     *     shamsys::instance::close();
     * }
     * ```
     */
    void init(int argc, char *argv[]);

    /// Start MPI
    void start_mpi(MPIInitInfo mpi_info);

    /// Start SYCL & MPI
    void init_sycl_mpi(std::string search_key, MPIInitInfo mpi_info);

    /**
     * @brief close the NodeInstance
     * Aka : Finalize both MPI & SYCL
     */
    void close();

    /// Finalize MPI
    void close_mpi();

    [[deprecated("Please use shamrock smi instead")]]
    void print_device_list();

    /// Print SYCL queue map
    void print_queue_map();

    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    /**
     * @brief
     *
     *
     *
     * @param id
     * @return sycl::queue& reference to the corresponding queue
     */
    sycl::queue &get_compute_queue(u32 id = 0);
    u32 get_compute_queue_eu_count(u32 id = 0);

    inline sycl::device get_compute_device() { return get_compute_queue().get_device(); }

    /**
     * @brief Get the alternative queue
     *
     * @param id
     * @return sycl::queue& reference to the corresponding queue
     */
    sycl::queue &get_alt_queue(u32 id = 0);

    sham::DeviceScheduler &get_compute_scheduler();
    sham::DeviceScheduler &get_alt_scheduler();

    std::shared_ptr<sham::DeviceScheduler> get_compute_scheduler_ptr();
    std::shared_ptr<sham::DeviceScheduler> get_alt_scheduler_ptr();

    ////////////////////////////
    // MPI related routines
    ////////////////////////////

    // idea to handle multiple GPU with MPI : i32 get_mpi_tag(u32 tag);

    void print_mpi_capabilities();

    void check_dgpu_available();

} // namespace shamsys::instance

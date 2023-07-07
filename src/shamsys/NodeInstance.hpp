// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeInstance.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file describing a Node Instance
 * @copyright Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
 *
 * A NodeInstance is a wrapper to perform a combined initialisation of both SYCL & MPI.
 * Essentially it handle the MPI process and corresponding accelerators
 */


#include "aliases.hpp"

#include <vector>
#include <sycl/sycl.hpp>

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

    /**
     * @brief initialize the NodeInstance from user inputs
     * ```
     * int main(int argc, char *argv[]){
     *     shamsys::instance::init(SyclInitInfo{alt_id, comp_id}, MPIInitInfo{argc, argv});
     *     ... do stuff ...
     *     shamsys::instance::close();
     * }
     * ```
     */
    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info);

    /**
     * @brief close the NodeInstance
     * Aka : Finalize both MPI & SYCL
     */
    void close();

    void print_device_list();

    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    /**
     * @brief Get the compute queue
     * 
     * @param id 
     * @return sycl::queue& reference to the corresponding queue
     */
    sycl::queue &get_compute_queue(u32 id = 0);

    inline sycl::device get_compute_device(){
        return get_compute_queue().get_device();
    }

    /**
     * @brief Get the alternative queue
     * 
     * @param id 
     * @return sycl::queue& reference to the corresponding queue
     */
    sycl::queue &get_alt_queue(u32 id = 0);

    ////////////////////////////
    // MPI related routines
    ////////////////////////////

    // idea to handle multiple GPU with MPI : i32 get_mpi_tag(u32 tag);

    /**
     * @brief the MPI world rank
     */
    inline u32 world_rank;

    /**
     * @brief the MPI world size
     */
    inline u32 world_size;

    /**
     * @brief Get the process name
     * @return std::string process name
     */
    std::string get_process_name();

} // namespace shamsys::instance
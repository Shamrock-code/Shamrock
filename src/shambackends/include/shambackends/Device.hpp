// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Device.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambackends/sycl.hpp"

namespace sham {

    enum class Vendor { UNKNOWN, NVIDIA, AMD, INTEL, APPLE };

    /**
     * @brief Returns the name of the given vendor
     *
     * @param v The vendor
     * @return The name of the given vendor
     */
    inline std::string vendor_name(Vendor v) {
        switch (v) {
        case Vendor::UNKNOWN: return "Unknown";
        case Vendor::NVIDIA: return "Nvidia";
        case Vendor::AMD: return "AMD";
        case Vendor::INTEL: return "Intel";
        case Vendor::APPLE: return "Apple";
        default:
            shambase::throw_unimplemented(
                "Unknown vendor"); // Throw an exception if the vendor is not recognized
        }
    }

    enum class Backend { UNKNOWN, CUDA, ROCM, OPENMP };

    /**
     * @brief Returns the name of the given backend
     *
     * @param b The backend
     * @return The name of the given backend
     * @throw shambase::unimplemented If the backend is not recognized
     */
    inline std::string backend_name(Backend b) {
        switch (b) {
        case Backend::UNKNOWN: return "Unknown";
        case Backend::CUDA: return "CUDA";
        case Backend::ROCM: return "ROCm";
        case Backend::OPENMP: return "OpenMP";
        default:
            shambase::throw_unimplemented(
                "Unknown backend"); // Throw an exception if the backend is not recognized
        }
    }

    /**
     * \brief Properties of a device
     *
     * This struct contains properties of a device, such as its vendor, backend,
     * amount of global memory and local memory, and cache size.
     */
    struct DeviceProperties {
        Vendor vendor;                    /**< The vendor of the device */
        Backend backend;                  /**< The backend of the device */
        usize global_mem_size;            /**< The amount of global memory on the device in bytes */
        usize global_mem_cache_line_size; /**< The size of the cache line used by the device in
                                             bytes */
        usize
            global_mem_cache_size; /**< The amount of global memory cache on the device in bytes */
        usize local_mem_size;      /**< The amount of shared local memory on the device in bytes */
    };

    struct DeviceMPIProperties {
        bool is_mpi_direct_capable;
    };

    /**
     * \brief Represents a SYCL device
     *
     * This class represents a SYCL device, which is a piece of hardware on
     * which kernels can be executed.
     */
    class Device {
        public:
        /**
         * \brief The id of the device
         */
        usize device_id;

        /**
         * \brief The SYCL device object
         */
        sycl::device dev;

        /**
         * \brief Properties of the device
         *
         * This struct contains properties of the device, such as its vendor,
         * backend, amount of global memory and local memory, and cache size.
         */
        DeviceProperties prop;

        /**
         * \brief Properties of the device regarding MPI
         */
        DeviceMPIProperties mpi_prop;

        /**
         * \brief Update the MPI properties of the device
         *
         * This function updates the MPI properties of the device based on its
         * capabilities.
         */
        void update_mpi_prop();

        /**
         * \brief Print info about the device
         *
         * This function prints information about the device, such as its id,
         * properties, and MPI capabilities.
         */
        void print_info();
    };

    std::vector<std::unique_ptr<Device>> get_device_list();

    Device sycl_dev_to_sham_dev(usize i, const sycl::device &dev);

} // namespace sham

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file queues.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/Device.hpp"
#include "shambase/string.hpp"

#include "shamcomm/mpiInfo.hpp"

namespace sham {

    /**
     * @brief Returns the type of backend of a SYCL device.
     * @param dev The SYCL device to query.
     * @return The backend of the given SYCL device.
     */
    Backend get_device_backend(const sycl::device &dev) {
        std::string pname = dev.get_platform().get_info<sycl::info::platform::name>();

        // The platform name may include information about the device
        // and/or the backend. We look for some keywords to determine
        // the backend.
        if (shambase::contain_substr(pname, "CUDA")) {
            return Backend::CUDA; // NVIDIA CUDA
        }
        if (shambase::contain_substr(pname, "NVIDIA")) {
            return Backend::CUDA;
        }
        if (shambase::contain_substr(pname, "ROCM")) {
            return Backend::ROCM; // AMD ROCm
        }
        if (shambase::contain_substr(pname, "AMD")) {
            return Backend::ROCM;
        }
        if (shambase::contain_substr(pname, "OpenMP")) {
            return Backend::OPENMP; // OpenMP
        }

        return Backend::UNKNOWN; // Unknown backend
    }

    /**
     * @brief Fetches the properties of a SYCL device.
     *
     * @param dev The SYCL device to query.
     * @return A structure containing the properties of the given
     *         SYCL device.
     */
    DeviceProperties fetch_properties(const sycl::device &dev) {
        return {
            Vendor::UNKNOWN,        // We cannot determine the vendor
            get_device_backend(dev) // Query the backend based on the platform name
        };
    }

    /**
     * @brief Fetches the MPI-related properties of a SYCL device.
     *
     * @param dev The SYCL device to query.
     * @param prop The properties of the device, as fetched using
     *             `fetch_properties()`.
     * @return A structure containing the MPI-related properties of the
     *         given SYCL device.
     */
    DeviceMPIProperties fetch_mpi_properties(const sycl::device &dev, DeviceProperties prop) {
        bool dgpu_capable = false;

        // If CUDA-aware MPI is enabled, and the device is a CUDA device,
        // then we can use it
        if ((shamcomm::mpi_cuda_aware == shamcomm::Yes) && (prop.backend == Backend::CUDA)) {
            dgpu_capable = true;
        }

        // Same for ROCm-aware MPI and ROCm devices
        if ((shamcomm::mpi_rocm_aware == shamcomm::Yes) && (prop.backend == Backend::ROCM)) {
            dgpu_capable = true;
        }

        // And for OpenMP since the data is on host is it by definition aware
        if (prop.backend == Backend::OPENMP) {
            dgpu_capable = true;
        }

        return DeviceMPIProperties{dgpu_capable};
    }

    /**
     * @brief Get a list of all SYCL devices
     *
     * This function returns a list of all SYCL devices available on
     * the system. Each device is identified by its unique SYCL id.
     *
     * @return A vector of SYCL devices
     */
    std::vector<sycl::device> get_sycl_device_list() {
        std::vector<sycl::device> devs; // The list of devices to be returned
        const auto &Platforms = sycl::platform::get_platforms();
        for (const auto &Platform : Platforms) {
            const auto &Devices = Platform.get_devices();
            for (const auto &Device : Devices) {
                devs.push_back(Device);
            }
        }
        return devs;
    }

    /**
     * @brief Convert a SYCL device to a shamrock backend device
     *
     * This function converts a SYCL device to a shamrock backend device.
     *
     * @param i The index of the device in the list of all devices
     * @param dev The SYCL device to be converted
     * @return A shamrock backend device corresponding to the given SYCL device
     */
    Device sycl_dev_to_sham_dev(usize i, const sycl::device &dev) {
        DeviceProperties prop       = fetch_properties(dev); // Get the properties of the device
        DeviceMPIProperties propmpi = fetch_mpi_properties(dev, prop); // Get the MPI properties
        return Device{
            i,      // The index of the device
            dev,    // The SYCL device
            prop,   // The properties of the device
            propmpi // The MPI properties of the device
        };
    }

    /**
     * @brief Get a list of all available devices
     *
     * This function returns a list of all available devices. The devices are
     * wrapped in a smart pointer and their index in the list is provided.
     *
     * @return A list of unique pointers to devices
     */
    std::vector<std::unique_ptr<Device>> get_device_list() {
        std::vector<sycl::device> devs = get_sycl_device_list();
        std::vector<std::unique_ptr<Device>> ret; // The return list of unique pointers to Device
        ret.reserve(devs.size());

        for (const sycl::device &dev : devs) {
            usize i = ret.size(); // Get the current index of the device
            ret.push_back(std::make_unique<Device>(sycl_dev_to_sham_dev(i, dev)));
        }

        return ret;
    }

    void Device::update_mpi_prop(){
        mpi_prop = fetch_mpi_properties(dev, prop);
    }

} // namespace sham
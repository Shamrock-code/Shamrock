// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Device.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/Device.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambackends/sysinfo.hpp"
#include "shamcomm/logs.hpp"
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
        if (shambase::contain_substr(pname, "HIP")) {
            return Backend::ROCM; // AMD ROCm
        }
        if (shambase::contain_substr(pname, "OpenMP")) {
            return Backend::OPENMP; // OpenMP
        }

        return Backend::UNKNOWN; // Unknown backend
    }

    /**
     * @brief Returns the type of a SYCL device.
     *
     * This function takes a SYCL device and returns a DeviceType enum that
     * represents the type of device. The type can be either CPU, GPU, or
     * UNKNOWN.
     *
     * @param dev The SYCL device to query.
     * @return A DeviceType enum that represents the type of device.
     */
    DeviceType get_device_type(const sycl::device &dev) {
        auto DeviceType = dev.get_info<sycl::info::device::device_type>();
        switch (DeviceType) {
        case sycl::info::device_type::cpu: return DeviceType::CPU;
        case sycl::info::device_type::gpu: return DeviceType::GPU;
        default: return DeviceType::UNKNOWN;
        }
    }

    /// Fetches a property of a SYCL device
#define FETCH_PROP(info_, info_type)                                                               \
    std::optional<info_type> info_ = [&]() -> std::optional<info_type> {                           \
        try {                                                                                      \
            return {dev.get_info<sycl::info::device::info_>()};                                    \
        } catch (...) {                                                                            \
            logger::warn_ln(                                                                       \
                "Device",                                                                          \
                "dev.get_info<sycl::info::device::" #info_ ">() raised an exception for device",   \
                name);                                                                             \
            return {};                                                                             \
        }                                                                                          \
    }();

    /// Fetches a property of a SYCL device (for cases where multiple prop would have the same name)
#define FETCH_PROPN(info_, info_type, n)                                                           \
    std::optional<info_type> n = [&]() -> std::optional<info_type> {                               \
        try {                                                                                      \
            return {dev.get_info<sycl::info::device::info_>()};                                    \
        } catch (...) {                                                                            \
            logger::warn_ln(                                                                       \
                "Device",                                                                          \
                "dev.get_info<sycl::info::device::" #info_ ">() raised an exception for device",   \
                name);                                                                             \
            return {};                                                                             \
        }                                                                                          \
    }();

    /**
     * @brief Fetches the properties of a SYCL device.
     *
     * @param dev The SYCL device to query.
     * @return A structure containing the properties of the given
     *         SYCL device.
     */
    DeviceProperties fetch_properties(const sycl::device &dev) {

        // Just to ensure that this one is not empty
        std::string name = "?";
        FETCH_PROPN(name, std::string, dev_name);
        if (dev_name) {
            name = *dev_name;
        }

        FETCH_PROP(vendor, std::string)

        FETCH_PROP(device_type, sycl::info::device_type)
        FETCH_PROP(vendor_id, uint32_t)
        FETCH_PROP(max_compute_units, uint32_t)
        FETCH_PROP(max_work_item_dimensions, uint32_t)
        FETCH_PROPN(max_work_item_sizes<1>, sycl::id<1>, max_work_item_sizes_1d)
        FETCH_PROPN(max_work_item_sizes<2>, sycl::id<2>, max_work_item_sizes_2d)
        FETCH_PROPN(max_work_item_sizes<3>, sycl::id<3>, max_work_item_sizes_3d)
        FETCH_PROP(max_work_group_size, size_t)
        FETCH_PROP(max_num_sub_groups, uint32_t)
        FETCH_PROP(sub_group_independent_forward_progress, bool)
        FETCH_PROP(sub_group_sizes, std::vector<size_t>)

        FETCH_PROP(preferred_vector_width_char, uint32_t)
        FETCH_PROP(preferred_vector_width_short, uint32_t)
        FETCH_PROP(preferred_vector_width_int, uint32_t)
        FETCH_PROP(preferred_vector_width_long, uint32_t)
        FETCH_PROP(preferred_vector_width_float, uint32_t)
        FETCH_PROP(preferred_vector_width_double, uint32_t)
        FETCH_PROP(preferred_vector_width_half, uint32_t)
        FETCH_PROP(native_vector_width_char, uint32_t)
        FETCH_PROP(native_vector_width_short, uint32_t)
        FETCH_PROP(native_vector_width_int, uint32_t)
        FETCH_PROP(native_vector_width_long, uint32_t)
        FETCH_PROP(native_vector_width_float, uint32_t)
        FETCH_PROP(native_vector_width_double, uint32_t)
        FETCH_PROP(native_vector_width_half, uint32_t)

        FETCH_PROP(max_clock_frequency, uint32_t)
        FETCH_PROP(address_bits, uint32_t)
        FETCH_PROP(max_mem_alloc_size, uint64_t)

        // Image a really second class objects in SYCL right now ...
        // FETCH_PROP(max_read_image_args, uint32_t)
        // FETCH_PROP(max_write_image_args, uint32_t)
        // FETCH_PROP(image2d_max_width, size_t)
        // FETCH_PROP(image2d_max_height, size_t)
        // FETCH_PROP(image3d_max_width, size_t)
        // FETCH_PROP(image3d_max_height, size_t)
        // FETCH_PROP(image3d_max_depth, size_t)
        // FETCH_PROP(image_max_buffer_size, size_t)
        // FETCH_PROP(max_samplers, uint32_t)

        FETCH_PROP(max_parameter_size, size_t)
        FETCH_PROP(mem_base_addr_align, uint32_t)
        FETCH_PROP(half_fp_config, std::vector<sycl::info::fp_config>)
        FETCH_PROP(single_fp_config, std::vector<sycl::info::fp_config>)
        FETCH_PROP(double_fp_config, std::vector<sycl::info::fp_config>)
        FETCH_PROP(global_mem_cache_type, sycl::info::global_mem_cache_type)
        FETCH_PROP(global_mem_cache_line_size, uint32_t)
        FETCH_PROP(global_mem_cache_size, uint64_t)
        FETCH_PROP(global_mem_size, uint64_t)
        FETCH_PROP(local_mem_type, sycl::info::local_mem_type)
        FETCH_PROP(local_mem_size, uint64_t)
        FETCH_PROP(error_correction_support, bool)
#ifdef SYCL_COMP_INTEL_LLVM
        FETCH_PROP(atomic_memory_order_capabilities, std::vector<sycl::memory_order>)
        FETCH_PROP(atomic_fence_order_capabilities, std::vector<sycl::memory_order>)
        FETCH_PROP(atomic_memory_scope_capabilities, std::vector<sycl::memory_scope>)
        FETCH_PROP(atomic_fence_scope_capabilities, std::vector<sycl::memory_scope>)
#endif
        FETCH_PROP(profiling_timer_resolution, size_t)
        FETCH_PROP(is_available, bool)
        FETCH_PROP(execution_capabilities, std::vector<sycl::info::execution_capability>)
        // FETCH_PROP(built_in_kernel_ids,std::vector<sycl::kernel_id>)
        // FETCH_PROP(built_in_kernels, std::vector<std::string>)
        // FETCH_PROP(platform, sycl::platform)

        FETCH_PROP(driver_version, std::string)
        FETCH_PROP(version, std::string)
#ifdef SYCL_COMP_INTEL_LLVM
        FETCH_PROP(backend_version, std::string)
#endif
        // FETCH_PROP(aspects, std::vector<sycl::aspect>)
        // FETCH_PROP(printf_buffer_size, size_t)
#ifdef SYCL_COMP_INTEL_LLVM
        // FETCH_PROP(parent_device, device)
#endif
        FETCH_PROP(partition_max_sub_devices, uint32_t)
        FETCH_PROP(partition_properties, std::vector<sycl::info::partition_property>)
        FETCH_PROP(partition_affinity_domains, std::vector<sycl::info::partition_affinity_domain>)
        FETCH_PROP(partition_type_property, sycl::info::partition_property)
        FETCH_PROP(partition_type_affinity_domain, sycl::info::partition_affinity_domain)

// On acpp 2^64-1 is returned, so we need to correct it
// see : https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1573
#ifdef SYCL_COMP_ACPP
        if (get_device_backend(dev) == Backend::OPENMP) {
            // Correct memory size
            auto physmem = sham::getPhysicalMemory();
            if (physmem) {
                global_mem_size = {*physmem};
            }
        }
#endif

        return {
            Vendor::UNKNOWN,         // We cannot determine the vendor
            get_device_backend(dev), // Query the backend based on the platform name
            get_device_type(dev),
            shambase::get_check_ref(global_mem_size),
            shambase::get_check_ref(global_mem_cache_line_size),
            shambase::get_check_ref(global_mem_cache_size),
            shambase::get_check_ref(local_mem_size),
            shambase::get_check_ref(max_compute_units)};
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
        if (shamcomm::is_direct_comm_aware(shamcomm::get_mpi_cuda_aware_status())
            && (prop.backend == Backend::CUDA)) {
            dgpu_capable = true;
        }

        // Same for ROCm-aware MPI and ROCm devices
        if (shamcomm::is_direct_comm_aware(shamcomm::get_mpi_rocm_aware_status())
            && (prop.backend == Backend::ROCM)) {
            dgpu_capable = true;
        }

        // And for OpenMP since the data is on host is it by definition aware
        if (prop.backend == Backend::OPENMP) {
            dgpu_capable = true;
        }

        // For other cases we can still force the DGPU state by setting a forced state
        if (auto forcing = shamcomm::should_force_dgpu_state()) {
            dgpu_capable = shamcomm::is_direct_comm_aware(*forcing);
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
        DeviceMPIProperties propmpi = {false};               // Get the MPI properties
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

    void Device::update_mpi_prop() { mpi_prop = fetch_mpi_properties(dev, prop); }

    void Device::print_info() {
        shamcomm::logs::raw_ln("  Device info :");
        switch (prop.backend) {
        case sham::Backend::OPENMP: shamcomm::logs::raw_ln("   - Backend : OpenMP"); break;
        case sham::Backend::CUDA: shamcomm::logs::raw_ln("   - Backend : CUDA"); break;
        case sham::Backend::ROCM: shamcomm::logs::raw_ln("   - Backend : ROCM"); break;
        case sham::Backend::UNKNOWN: shamcomm::logs::raw_ln("   - Backend : Unknown"); break;
        }
        switch (prop.vendor) {
        case sham::Vendor::AMD: shamcomm::logs::raw_ln("   - Vendor : AMD"); break;
        case sham::Vendor::APPLE: shamcomm::logs::raw_ln("   - Vendor : Apple"); break;
        case sham::Vendor::INTEL: shamcomm::logs::raw_ln("   - Vendor : Intel"); break;
        case sham::Vendor::NVIDIA: shamcomm::logs::raw_ln("   - Vendor : Nvidia"); break;
        case sham::Vendor::UNKNOWN: shamcomm::logs::raw_ln("   - Vendor : Unknown"); break;
        }
        logger::raw_ln("   - Global mem size :", shambase::readable_sizeof(prop.global_mem_size));
        logger::raw_ln(
            "   - Cache line size :", shambase::readable_sizeof(prop.global_mem_cache_line_size));
        logger::raw_ln(
            "   - Cache size      :", shambase::readable_sizeof(prop.global_mem_cache_size));
        logger::raw_ln("   - Local mem size  :", shambase::readable_sizeof(prop.local_mem_size));
        logger::raw_ln("   - Direct MPI capable :", mpi_prop.is_mpi_direct_capable);
    }

} // namespace sham

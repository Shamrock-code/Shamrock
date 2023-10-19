// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
/**
 * @file sycl.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
#ifdef SHAMROCK_ENABLE_BACKEND_SYCL

#include "sycl.hpp"


namespace sham::details {


    namespace handle {

        std::unique_ptr<sycl::device> compute_device;
        std::unique_ptr<sycl::device> host_device;

        std::unique_ptr<sycl::queue> device_handle;
        std::unique_ptr<sycl::queue> host_handle;

        DeviceContextNative get_device() {

            return DeviceContextNative{*compute_device, *device_handle};
        }

        HostContextNative get_host() {

            return HostContextNative{*host_device, *host_handle};
        }

    } // namespace handle

    void backend_initialize(int argc, char *argv[]) {

        


    }

    void backend_finalize() { 
        handle::device_handle.reset();
        handle::host_handle.reset();
        handle::compute_device.reset();
        handle::host_device.reset();
    }

} // namespace sham::details

#endif
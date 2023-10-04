// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#ifdef SHAMROCK_ENABLE_BACKEND_SYCL

#include "aliases/basetypes.hpp"
#include "aliases/half.hpp"
#include "aliases/vec.hpp"
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

enum SYCLImplementation { OPENSYCL, DPCPP, UNKNOWN };

#ifdef SYCL_COMP_ACPP
constexpr SYCLImplementation sycl_implementation = OPENSYCL;
#else
    #ifdef SYCL_COMP_INTEL_LLVM
constexpr SYCLImplementation sycl_implementation = DPCPP;
    #else
constexpr SYCLImplementation sycl_implementation = UNKNOWN;
    #endif
#endif

namespace sham::details {

    struct DeviceStreamNative {
        sycl::queue &queue_obj;
    };

    struct HostStreamNative {
        sycl::queue &queue_obj;
    };

    struct DeviceContextNative {
        sycl::device &device_obj;
        sycl::queue &queue_obj;

        inline DeviceStreamNative get_stream(u32 i = 0) { return DeviceStreamNative{queue_obj}; }
    };

    struct HostContextNative {
        sycl::device &device_obj;
        sycl::queue &queue_obj;
        inline HostStreamNative get_stream(u32 i = 0) { return HostStreamNative{queue_obj}; }
    };

    namespace handle {

        inline std::unique_ptr<sycl::device> compute_device;
        inline std::unique_ptr<sycl::device> host_device;

        inline std::unique_ptr<sycl::queue> device_handle;
        inline std::unique_ptr<sycl::queue> host_handle;

        inline DeviceContextNative get_device() {

            return DeviceContextNative{*compute_device, *device_handle};
        }

        inline HostContextNative get_host() {

            return HostContextNative{*host_device, *host_handle};
        }

    } // namespace handle

    inline void backend_initialize(int argc, char *argv[]) {

        


    }

    inline void backend_finalize() { 
        handle::device_handle.reset();
        handle::host_handle.reset();
        handle::compute_device.reset();
        handle::host_device.reset();
    }

} // namespace sham::details

#endif
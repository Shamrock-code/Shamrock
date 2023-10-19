// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
/**
 * @file sycl.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
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

        DeviceContextNative get_device();

        HostContextNative get_host();

    } // namespace handle

    void backend_initialize(int argc, char *argv[]);

    void backend_finalize();

} // namespace sham::details

#endif
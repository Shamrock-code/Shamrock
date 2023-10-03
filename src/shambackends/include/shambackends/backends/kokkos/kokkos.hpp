// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "check_backend.hpp"

namespace sham::details {

 static constexpr bool multiple_device_support = false;

    struct DeviceStreamNative {
        
    };

    struct HostStreamNative {
        
    };

    struct DeviceContextNative {
        

        inline DeviceStreamNative get_stream(u32 i = 0) { return DeviceStreamNative{queue_obj}; }
    };

    struct HostContextNative {
        
        inline HostStreamNative get_stream(u32 i = 0) { return HostStreamNative{queue_obj}; }
    };

    namespace handle {



        inline DeviceContextNative get_device(u32 i = 0) {
            if (i != 0) {
                throw std::invalid_argument("The sycl backend does not support multiple device, "
                                            "please call get_device only with i=0");
            }
            return DeviceContextNative{*compute_device, *device_handle};
        }

        inline HostContextNative get_host(u32 i = 0) {
            if (i != 0) {
                throw std::invalid_argument("The sycl backend does not support multiple host, "
                                            "please call get_host only with i=0");
            }
            return HostContextNative{*host_device, *host_handle};
        }

    } // namespace handle

    void initialize_backend() {}

    void finalize_backend() {}

} // namespace sham::details
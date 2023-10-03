// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "check_backend.hpp"

#include "aliases/basetypes.hpp"

namespace sham::details {

 static constexpr bool multiple_device_support = false;

    struct DeviceStreamNative {
        
    };

    struct HostStreamNative {
        
    };

    struct DeviceContextNative {
        

        inline DeviceStreamNative get_stream(u32 i = 0) { return DeviceStreamNative{}; }
    };

    struct HostContextNative {
        
        inline HostStreamNative get_stream(u32 i = 0) { return HostStreamNative{}; }
    };

    namespace handle {



        inline DeviceContextNative get_device(u32 i = 0) {
            if (i != 0) {
                throw std::invalid_argument("The sycl backend does not support multiple device, "
                                            "please call get_device only with i=0");
            }
            return DeviceContextNative{};
        }

        inline HostContextNative get_host(u32 i = 0) {
            if (i != 0) {
                throw std::invalid_argument("The sycl backend does not support multiple host, "
                                            "please call get_host only with i=0");
            }
            return HostContextNative{};
        }

    } // namespace handle

    inline void backend_initialize(int argc, char *argv[]) {

        


    }

    inline void backend_finalize() { 
        
    }

} // namespace sham::details
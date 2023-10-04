// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#ifdef SHAMROCK_ENABLE_BACKEND_KOKKOS

#include "Kokkos_Core.hpp"
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



        inline DeviceContextNative get_device() {
            return DeviceContextNative{};
        }

        inline HostContextNative get_host() {
            return HostContextNative{};
        }

    } // namespace handle

    inline void backend_initialize(int argc, char *argv[]) {

        Kokkos::initialize(argc,argv);        


    }

    inline void backend_finalize() { 
        Kokkos::finalize();
    }

} // namespace sham::details

#endif
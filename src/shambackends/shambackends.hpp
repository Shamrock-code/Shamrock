// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
/**
 * @file shambackends.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
#include <utility>

#include <shambackends/backends/sycl/sycl.hpp>
#include <shambackends/backends/kokkos/kokkos.hpp>

namespace sham {

    struct DeviceStream {
        using InternalHndl = details::DeviceStreamNative;

        InternalHndl hndl;

        public:
        DeviceStream(InternalHndl hndl) : hndl(std::move(hndl)) {}
    };

    struct HostStream {
        using InternalHndl = details::HostStreamNative;

        InternalHndl hndl;

        public:
        HostStream(InternalHndl hndl) : hndl(std::move(hndl)) {}
    };

    struct DeviceContext {
        using InternalHndl = details::DeviceContextNative;

        InternalHndl hndl;

        public:
        DeviceContext(InternalHndl hndl) : hndl(std::move(hndl)) {}

        inline DeviceStream get_stream(u32 i = 0) { return hndl.get_stream(i); }
    };

    struct HostContext {
        using InternalHndl = details::HostContextNative;

        InternalHndl hndl;

        public:
        HostContext(InternalHndl hndl) : hndl(std::move(hndl)) {}

        inline HostStream get_stream(u32 i = 0) { return hndl.get_stream(i); }
    };

    inline DeviceContext get_device() { return details::handle::get_device(); }

    inline HostContext get_host() { return details::handle::get_host(); }

    
    inline void backend_initialize(int argc, char *argv[]) { details::backend_initialize(argc, argv); }

    inline void backend_finalize() { details::backend_finalize(); }

} // namespace sham
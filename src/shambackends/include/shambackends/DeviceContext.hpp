// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#ifdef SHAMBACKENDS_USE_SYCL
    #include <shambackends/backends/sycl/DeviceContext.hpp>
    #include <shambackends/backends/sycl/sycl.hpp>
#endif

namespace sham {

    class DeviceContext {
        using InternalsHndl = details::DeviceContextNative;

        InternalsHndl hndl;

        public:
        inline InternalsHndl &get_native_handle() { return hndl; }
    };

} // namespace sham

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "DeviceContext.hpp"
#ifdef SHAMBACKENDS_USE_SYCL
    #include <shambackends/backends/sycl/DeviceQueue.hpp>
    #include <shambackends/backends/sycl/sycl.hpp>
#endif

namespace sham {

    enum DeviceQueueSchedulingType { GRAPH, STREAM, IN_ORDER };

    class DeviceQueue {
        using InternalsHndl = details::DeviceQueueNative;

        InternalsHndl hndl;

        public:
        DeviceQueue(DeviceContext &ctx) : hndl(ctx.get_native_handle()) {}

        inline InternalsHndl &get_native_handle() { return hndl; }
    };

} // namespace sham

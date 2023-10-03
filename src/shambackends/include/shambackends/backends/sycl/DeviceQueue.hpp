// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "check_backend.hpp"

#include "shambackends/backends/sycl/DeviceContext.hpp"

namespace sham::details {

    struct DeviceQueueNative {
        std::unique_ptr<sycl::queue> q;
        explicit DeviceQueueNative(DeviceContextNative &ctx) {
            q = std::make_unique<sycl::queue>(ctx.device_obj);
        }
    };

} // namespace sham::details

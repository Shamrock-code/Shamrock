// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceContext.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/Device.hpp"

namespace sham {

    class DeviceContext {
        public:
        std::shared_ptr<Device> device;

        sycl::context ctx;

        void print_info();

        DeviceContext(std::shared_ptr<Device> device);
    };

} // namespace sham
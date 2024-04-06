// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceQueue.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceContext.hpp"

namespace sham {

    class DeviceQueue {
        public:
        std::shared_ptr<DeviceContext> ctx;

        sycl::queue q;

        std::string queue_name;
        bool in_order;

        void test();

        DeviceQueue(std::string queue_name, std::shared_ptr<DeviceContext> ctx, bool in_order);
    };

} // namespace sham
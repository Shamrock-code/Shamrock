// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RadixTree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceQueue.hpp"

namespace sham {

    class DeviceScheduler {
        public:
        DeviceContext *ctx;

        std::vector<std::unique_ptr<DeviceQueue>> queues;

        DeviceScheduler(DeviceContext *ctx);

        DeviceQueue &get_queue(u32 id = 0);

        void print_info();
    };

} // namespace sham
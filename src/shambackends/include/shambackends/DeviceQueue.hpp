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

#include "shambackends/DeviceContext.hpp"

namespace sham {

    class DeviceQueue{
        public:

        DeviceContext * ctx;

        sycl::queue q;

        std::string queue_name;
        bool in_order;

        DeviceQueue(std::string queue_name, DeviceContext * ctx, bool in_order);

    };

}
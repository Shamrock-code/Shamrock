// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file device_select.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/Device.hpp"
#include <string>

namespace shamsys {

    struct DeviceSelectRet_t {

        std::shared_ptr<sham::Device> device_compute;
        std::shared_ptr<sham::Device> device_alt;
    };

    DeviceSelectRet_t select_devices(std::string sycl_cfg);

} // namespace shamsys

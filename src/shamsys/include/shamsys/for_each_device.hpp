// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file for_each_device.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include <functional>

namespace shamsys {

    /**
     * @brief for each SYCL device
     *
     * @param fct
     * @return u32 the number of devices
     */
    inline u32
    for_each_device(std::function<void(u32, const sycl::platform &, const sycl::device &)> fct) {

        u32 key_global        = 0;
        const auto &Platforms = sycl::platform::get_platforms();
        for (const auto &Platform : Platforms) {
            const auto &Devices = Platform.get_devices();
            for (const auto &Device : Devices) {
                fct(key_global, Platform, Device);
                key_global++;
            }
        }
        return key_global;
    }
} // namespace shamsys

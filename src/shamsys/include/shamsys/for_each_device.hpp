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
 * @file for_each_device.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include <functional>

namespace shamsys {

    /**
     * @brief Iterate over all SYCL devices and perform a given function.
     *
     * @param fct The function to be called for each device. The function takes 3 arguments:
     * - The key of the device. This key is a unique identifier for the device.
     * - The SYCL platform corresponding to the device.
     * - The SYCL device.
     *
     * @return The total number of devices found.
     *
     * @details
     * The function will be called in the order of the platforms and devices.
     * The order of the platforms is determined by the SYCL implementation.
     * The order of the devices is determined by the order of the platforms and the order of the
     * devices within each platform.
     *
     * Example usage:
     *
     * @code
     * auto fct = [&](u32 key, const sycl::platform &Platform, const sycl::device &Device) {
     *     std::cout << "Platform: " << Platform.get_info<sycl::info::platform::name>() <<
     * std::endl; std::cout << "Device: " << Device.get_info<sycl::info::device::name>() <<
     * std::endl;
     * };
     * shamsys::for_each_device(fct);
     * @endcode
     */
    inline u32 for_each_device(
        std::function<void(u32, const sycl::platform &, const sycl::device &)> fct) {

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

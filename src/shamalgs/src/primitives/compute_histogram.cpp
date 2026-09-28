// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file compute_histogram.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the implementation selector of compute_histogram.
 */

#include "shamalgs/primitives/compute_histogram.hpp"

namespace shamalgs::primitives::impl {

    ComputeHistogramImpl compute_histogram_impl{
        [](const sham::DeviceScheduler_ptr &dev_sched, auto &self) {
            if (dev_sched->ctx->device->prop.type == sham::DeviceType::GPU) {
                self.set(GpuOversubscribe{});
            } else {
                self.set(NaiveGpu{}); // it is portable and fast everywhere
            }
        }};

} // namespace shamalgs::primitives::impl

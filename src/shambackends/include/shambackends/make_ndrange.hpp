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
 * @file make_ndrange.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp"

namespace sham {

    /// Builds an `nd_range` with work-group size `wg_size`, rounding `nthread` up to the next
    /// multiple of `wg_size` so every work-group launched is full.
    inline sycl::nd_range<1> make_ndrange(u32 wg_size, u32 nthread) {
        u32 nthread_rounded = ((nthread + wg_size - 1) / wg_size) * wg_size;
        return sycl::nd_range<1>(sycl::range<1>(nthread_rounded), sycl::range<1>(wg_size));
    }

} // namespace sham

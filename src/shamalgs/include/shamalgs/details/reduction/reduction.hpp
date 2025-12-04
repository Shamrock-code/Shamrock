// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file reduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction {

    template<class T>
    bool has_nan(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

    template<class T>
    bool has_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

    template<class T>
    bool has_nan_or_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

} // namespace shamalgs::reduction

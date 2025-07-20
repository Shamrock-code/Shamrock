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
 * @file bitonicSort_updated_usm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief main include file for the shamalgs algorithms
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"

/**
 * @brief namespace to store algorithms implemented by shamalgs
 *
 */
namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated_usm(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len);

} // namespace shamalgs::algorithm::details

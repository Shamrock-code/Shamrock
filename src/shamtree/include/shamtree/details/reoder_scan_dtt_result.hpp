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
 * @file reoder_scan_dtt_result.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamtree::details {

    inline void reorder_scan_dtt_result(
        u32 N, sham::DeviceBuffer<u32_2> &in_out, sham::DeviceBuffer<u32> &offsets) {

        size_t interact_count = in_out.get_size();
        size_t offsets_count  = N + 1;

        offsets.resize(offsets_count);
        offsets.fill(0);

        auto &q = in_out.get_dev_scheduler().get_queue();

        // very brutal way of atomic counting the number of interactions for each sender
        sham::kernel_call(
            q,
            sham::MultiRef{in_out},
            sham::MultiRef{offsets},
            interact_count,
            [N](u32 i, const u32_2 *__restrict__ in_out, u32 *__restrict__ offsets) {
                SHAM_ASSERT(in_out[i].x() < N);

                sycl::atomic_ref<
                    u32,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    atom(offsets[in_out[i].x()]);
                atom += 1_u32;
            });

        shamalgs::primitives::scan_exclusive_sum_in_place(offsets, offsets_count);

        // here we can global sort in_out, or atomic store then local sort, for now i do a CPU sort
        // for testing
        std::vector<u32_2> in_out_stdvec = in_out.copy_to_stdvec();
        std::sort(in_out_stdvec.begin(), in_out_stdvec.end(), [](u32_2 a, u32_2 b) {
            return a.x() < b.x();
        });
        in_out.copy_from_stdvec(in_out_stdvec);
    }

} // namespace shamtree::details

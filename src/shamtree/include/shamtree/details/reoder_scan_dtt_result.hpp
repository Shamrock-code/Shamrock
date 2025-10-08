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

        // here we can global sort in_out, or atomic store then local sort,
        // for now i do a CPU sort for testing
        if (true) {
            sham::DeviceBuffer<u32_2> in_out_sorted(
                in_out.get_size(), in_out.get_dev_scheduler_ptr());

            sham::DeviceBuffer<u32> offset2 = offsets.copy();

            // here we do a global sort by atomic fetch add on first index. The result is not yet
            // deterministic since it depends on threads execution order.
            sham::kernel_call(
                q,
                sham::MultiRef{in_out},
                sham::MultiRef{in_out_sorted, offset2},
                interact_count,
                [N](u32 i,
                    const u32_2 *__restrict__ in_out,
                    u32_2 *__restrict__ in_out_sorted,
                    u32 *__restrict__ local_head) {
                    SHAM_ASSERT(in_out[i].x() < N);

                    sycl::atomic_ref<
                        u32,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        atom(local_head[in_out[i].x()]);

                    u32 ret = atom.fetch_add(1_u32);

                    in_out_sorted[ret] = in_out[i];
                });

            // we now perform a local sort on each slots which make the result deterministic
            sham::kernel_call(
                q,
                sham::MultiRef{offsets},
                sham::MultiRef{in_out_sorted},
                N,
                [N](u32 gid, const u32 *__restrict__ offsets, u32_2 *__restrict__ in_out_sorted) {
                    u32 start_index = offsets[gid];
                    u32 end_index   = offsets[gid + 1];

                    SHAM_ASSERT(start_index < end_index);
                    SHAM_ASSERT(start_index < N);
                    SHAM_ASSERT(end_index <= N);

                    auto comp = [](u32_2 a, u32_2 b) {
                        return (a.x() == b.x()) ? (a.y() < b.y()) : (a.x() < b.x());
                    };

                    // simple insertion sort between those indexes
                    for (int i = start_index + 1; i < end_index; ++i) {
                        auto key = in_out_sorted[i];
                        int j    = i - 1;
                        while (j >= start_index && comp(key, in_out_sorted[j])) {
                            in_out_sorted[j + 1] = in_out_sorted[j];
                            --j;
                        }
                        in_out_sorted[j + 1] = key;
                    }
                });

            in_out = std::move(in_out_sorted);
        } else {

            std::vector<u32_2> in_out_stdvec = in_out.copy_to_stdvec();
            std::sort(in_out_stdvec.begin(), in_out_stdvec.end(), [](u32_2 a, u32_2 b) {
                return a.x() < b.x();
            });
            in_out.copy_from_stdvec(in_out_stdvec);
        }
    }

} // namespace shamtree::details

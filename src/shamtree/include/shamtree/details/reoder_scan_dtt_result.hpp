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
#include "shambase/stacktrace.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamtree::details {

    inline void reorder_scan_dtt_result(
        u32 N, sham::DeviceBuffer<u32_2> &in_out, sham::DeviceBuffer<u32> &offsets) {

        __shamrock_stack_entry();

        size_t interact_count = in_out.get_size();
        size_t offsets_count  = N + 1;

        offsets.resize(offsets_count);
        offsets.fill(0);

        if (in_out.get_size() == 0) {
            return; // no kernel call if there is no interaction, but we still need to return an
                    // offset table that is [0,0]
        }

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
                [interact_count](
                    u32 gid, const u32 *__restrict__ offsets, u32_2 *__restrict__ in_out_sorted) {
                    u32 start_index = offsets[gid];
                    u32 end_index   = offsets[gid + 1];

                    // can be equal if there is no interaction for this sender
                    SHAM_ASSERT(start_index <= end_index);

                    // skip empty ranges to avoid unnecessary work
                    if (start_index == end_index) {
                        return;
                    }

                    // if there is no interactions at the end of the offset list
                    // offsets[gid] can be equal to interact_count
                    // but we check that start_index != end_index, so here the correct assertions
                    // is indeed start_index < interact_count
                    SHAM_ASSERT(start_index < interact_count);
                    SHAM_ASSERT(end_index <= interact_count); // see the for loop for this one

                    auto comp = [](u32_2 a, u32_2 b) {
                        return (a.x() == b.x()) ? (a.y() < b.y()) : (a.x() < b.x());
                    };

                    // simple insertion sort between those indexes
                    for (u32 i = start_index + 1; i < end_index; ++i) {
                        auto key = in_out_sorted[i];
                        u32 j    = i;
                        while (j > start_index && comp(key, in_out_sorted[j - 1])) {
                            in_out_sorted[j] = in_out_sorted[j - 1];
                            --j;
                        }
                        in_out_sorted[j] = key;
                    }
                });

            in_out = std::move(in_out_sorted);
        } else {

            std::vector<u32_2> in_out_stdvec = in_out.copy_to_stdvec();
            std::sort(in_out_stdvec.begin(), in_out_stdvec.end(), [](u32_2 a, u32_2 b) {
                return (a.x() == b.x()) ? (a.y() < b.y()) : (a.x() < b.x());
            });
            in_out.copy_from_stdvec(in_out_stdvec);
        }
    }

} // namespace shamtree::details

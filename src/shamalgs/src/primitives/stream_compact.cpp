// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sort_by_keys.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shamalgs/primitives/stream_compact.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    sham::DeviceBuffer<u32> stream_compact(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &&buf_flags, u32 len) {

        if (buf_flags.get_size() < len + 1)
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "buf_flags.get_size() < len+1\n  buf_flags.get_size() = {}, len = {}",
                buf_flags.get_size(),
                len));

        shamalgs::primitives::scan_exclusive_sum_in_place(buf_flags, len + 1);

        u32 new_len = buf_flags.get_val_at_idx(len);

        sham::DeviceBuffer<u32> index_map(new_len, sched);

        if (new_len > 0) {
            sham::kernel_call(
                sched->get_queue(),
                sham::MultiRef{buf_flags},
                sham::MultiRef{index_map},
                len + 1,
                [](u32 idx, const u32 *sum_vals, u32 *new_idx) {
                    u32 current_val = sum_vals[idx];

                    bool should_write = (current_val < sum_vals[idx + 1]);

                    if (should_write) {
                        new_idx[current_val] = idx;
                    }
                });
        }

        return index_map;
    }

} // namespace shamalgs::primitives

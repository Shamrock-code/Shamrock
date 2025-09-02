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
 * @file groupReduction_usm_impl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction::details {

    template<class T, class GroupCombiner, class IdentityGetter>
    inline sycl::event reduc_step(
        sham::DeviceQueue &q,
        T *global_mem,
        sham::EventList &depends_list,
        u32 len,
        u32 &cur_slice_sz,
        u32 &remaining_val,
        u32 work_group_size,
        GroupCombiner &&group_combine,
        IdentityGetter &&identity_getter) {

        sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            u32 slice_read_size  = cur_slice_sz;
            u32 slice_write_size = cur_slice_sz * work_group_size;
            u32 max_id           = len;

            cgh.parallel_for(exec_range, [=](sycl::nd_item<1> item) {
                u64 lid           = item.get_local_id(0);
                u64 group_tile_id = item.get_group_linear_id();
                u64 gid           = group_tile_id * work_group_size + lid;

                u64 iread  = gid * slice_read_size;
                u64 iwrite = group_tile_id * slice_write_size;

                T val_read = (iread < max_id) ? global_mem[iread] : identity_getter();

                T local_red = group_combine(item.get_group(), val_read);

                // can be removed if i change the index in the look back ?
                if (lid == 0) {
                    global_mem[iwrite] = local_red;
                }
            });
        });

        cur_slice_sz *= work_group_size;
        remaining_val = exec_range.get_group_range().size();

        return e;
    }

    template<class T, class GroupCombiner, class BinaryOp, class IdentityGetter>
    inline T reduc_internal(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size,
        GroupCombiner &&group_combine,
        BinaryOp &&binary_op,
        IdentityGetter &&identity_getter) {

        sham::DeviceQueue &q = shambase::get_check_ref(sched).get_queue();

        if (start_id >= end_id) {
            shambase::throw_with_loc<std::invalid_argument>(
                "Empty (or invalid) range not supported for reduction operation");
        }

        u32 len = end_id - start_id;

        sham::DeviceBuffer<T> buf_int(len, sched);

        buf1.copy_range(start_id, end_id, buf_int);

        sham::EventList depends_list;
        T *compute_buf = buf_int.get_write_access(depends_list);

        u32 cur_slice_sz  = 1;
        u32 remaining_val = len;
        while (len / cur_slice_sz > work_group_size * 8) {
            auto e = reduc_step<T>(
                q,
                compute_buf,
                depends_list,
                len,
                cur_slice_sz,
                remaining_val,
                work_group_size,
                std::forward<GroupCombiner>(group_combine),
                std::forward<IdentityGetter>(identity_getter));

            sham::EventList old_list;
            std::swap(depends_list, old_list);
            depends_list.add_event(e);
        }

        sham::DeviceBuffer<T> recov_buf(remaining_val, sched);
        T *result = recov_buf.get_write_access(depends_list);

        sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);
        auto e = q.submit(depends_list, [&, remaining_val](sycl::handler &cgh) {
            u32 slice_read_size = cur_slice_sz;

            cgh.parallel_for(exec_range, [=](sycl::nd_item<1> item) {
                u64 lid           = item.get_local_id(0);
                u64 group_tile_id = item.get_group_linear_id();
                u64 gid           = group_tile_id * work_group_size + lid;

                u64 iread = gid * slice_read_size;

                if (gid >= remaining_val) {
                    return;
                }

                result[gid] = compute_buf[iread];
            });
        });

        buf_int.complete_event_state(e);
        recov_buf.complete_event_state(e);

        auto acc = recov_buf.copy_to_stdvec();
        T ret    = acc[0]; // init value
        for (u64 i = 1; i < remaining_val; i++) {
            ret = binary_op(ret, acc[i]);
        }

        return ret;
    }
} // namespace shamalgs::reduction::details

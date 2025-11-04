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
 * @file groupReduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/group_op.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

template<class T, u32 work_group_size>
class KernelSliceReduceSum;

template<class T, u32 work_group_size>
class KernelSliceReduceMin;

template<class T, u32 work_group_size>
class KernelSliceReduceMax;

namespace shamalgs::reduction::details {

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

    template<class T, u32 work_group_size>
    struct GroupReduction {

        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);
    };

    template<class T, u32 work_group_size>
    inline T GroupReduction<T, work_group_size>::sum(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);

        shamalgs::memory::write_with_offset_into(q, buf_int, buf1, start_id, len);

        u32 cur_slice_sz  = 1;
        u32 remaining_val = len;
        while (len / cur_slice_sz > work_group_size * 8) {

            sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor global_mem{buf_int, cgh, sycl::read_write};

                u32 slice_read_size  = cur_slice_sz;
                u32 slice_write_size = cur_slice_sz * work_group_size;
                u32 max_id           = len;

                cgh.parallel_for<KernelSliceReduceSum<T, work_group_size>>(
                    exec_range, [=](sycl::nd_item<1> item) {
                        u64 lid           = item.get_local_id(0);
                        u64 group_tile_id = item.get_group_linear_id();
                        u64 gid           = group_tile_id * work_group_size + lid;

                        u64 iread  = gid * slice_read_size;
                        u64 iwrite = group_tile_id * slice_write_size;

                        T val_read = (iread < max_id) ? global_mem[iread]
                                                      : shambase::VectorProperties<T>::get_zero();

                        T local_red = sham::sum_over_group(item.get_group(), val_read);

                        // can be removed if i change the index in the look back ?
                        if (lid == 0) {
                            global_mem[iwrite] = local_red;
                        }
                    });
            });

            cur_slice_sz *= work_group_size;
            remaining_val = exec_range.get_group_range().size();
        }

        sycl::buffer<T> recov{remaining_val};

        sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);
        q.submit([&, remaining_val](sycl::handler &cgh) {
            sycl::accessor compute_buf{buf_int, cgh, sycl::read_only};
            sycl::accessor result{recov, cgh, sycl::write_only, sycl::no_init};

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

        T ret = shambase::VectorProperties<T>::get_zero();
        {
            sycl::host_accessor acc{recov, sycl::read_only};
            for (u64 i = 0; i < remaining_val; i++) {
                ret += acc[i];
            }
        }

        return ret;
    }

    template<class T, u32 work_group_size>
    inline T GroupReduction<T, work_group_size>::min(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);

        shamalgs::memory::write_with_offset_into(q, buf_int, buf1, start_id, len);

        u32 cur_slice_sz  = 1;
        u32 remaining_val = len;
        while (len / cur_slice_sz > work_group_size * 8) {

            sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor global_mem{buf_int, cgh, sycl::read_write};

                u32 slice_read_size  = cur_slice_sz;
                u32 slice_write_size = cur_slice_sz * work_group_size;
                u32 max_id           = len;

                cgh.parallel_for<KernelSliceReduceMin<T, work_group_size>>(
                    exec_range, [=](sycl::nd_item<1> item) {
                        u64 lid           = item.get_local_id(0);
                        u64 group_tile_id = item.get_group_linear_id();
                        u64 gid           = group_tile_id * work_group_size + lid;

                        u64 iread  = gid * slice_read_size;
                        u64 iwrite = group_tile_id * slice_write_size;

                        T val_read = (iread < max_id) ? global_mem[iread]
                                                      : shambase::VectorProperties<T>::get_max();

                        T local_red = sham::min_over_group(item.get_group(), val_read);

                        // can be removed if i change the index in the look back ?
                        if (lid == 0) {
                            global_mem[iwrite] = local_red;
                        }
                    });
            });

            cur_slice_sz *= work_group_size;
            remaining_val = exec_range.get_group_range().size();
        }

        sycl::buffer<T> recov{remaining_val};

        sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);
        q.submit([&, remaining_val](sycl::handler &cgh) {
            sycl::accessor compute_buf{buf_int, cgh, sycl::read_only};
            sycl::accessor result{recov, cgh, sycl::write_only, sycl::no_init};

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

        T ret = shambase::VectorProperties<T>::get_max();
        {
            sycl::host_accessor acc{recov, sycl::read_only};
            for (u64 i = 0; i < remaining_val; i++) {
                ret = sham::min(acc[i], ret);
            }
        }

        return ret;
    }

    template<class T, u32 work_group_size>
    inline T GroupReduction<T, work_group_size>::max(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);

        shamalgs::memory::write_with_offset_into(q, buf_int, buf1, start_id, len);

        u32 cur_slice_sz  = 1;
        u32 remaining_val = len;
        while (len / cur_slice_sz > work_group_size * 8) {

            sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor global_mem{buf_int, cgh, sycl::read_write};

                u32 slice_read_size  = cur_slice_sz;
                u32 slice_write_size = cur_slice_sz * work_group_size;
                u32 max_id           = len;

                cgh.parallel_for<KernelSliceReduceMax<T, work_group_size>>(
                    exec_range, [=](sycl::nd_item<1> item) {
                        u64 lid           = item.get_local_id(0);
                        u64 group_tile_id = item.get_group_linear_id();
                        u64 gid           = group_tile_id * work_group_size + lid;

                        u64 iread  = gid * slice_read_size;
                        u64 iwrite = group_tile_id * slice_write_size;

                        T val_read = (iread < max_id) ? global_mem[iread]
                                                      : shambase::VectorProperties<T>::get_min();

                        T local_red = sham::max_over_group(item.get_group(), val_read);

                        // can be removed if i change the index in the look back ?
                        if (lid == 0) {
                            global_mem[iwrite] = local_red;
                        }
                    });
            });

            cur_slice_sz *= work_group_size;
            remaining_val = exec_range.get_group_range().size();
        }

        sycl::buffer<T> recov{remaining_val};

        sycl::nd_range<1> exec_range = shambase::make_range(remaining_val, work_group_size);
        q.submit([&, remaining_val](sycl::handler &cgh) {
            sycl::accessor compute_buf{buf_int, cgh, sycl::read_only};
            sycl::accessor result{recov, cgh, sycl::write_only, sycl::no_init};

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

        T ret = shambase::VectorProperties<T>::get_min();
        {
            sycl::host_accessor acc{recov, sycl::read_only};
            for (u64 i = 0; i < remaining_val; i++) {
                ret = sham::max(acc[i], ret);
            }
        }

        return ret;
    }
#endif

} // namespace shamalgs::reduction::details

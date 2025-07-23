// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonKernels.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/integer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shamtree/MortonKernels.hpp"

template<class T>
class fill_trailling_buf;

template<class morton_t, class pos_t, u32 dim>
class pos_to_morton;

template<class morton_t, class pos_t, u32 dim>
class pos_to_morton_usm;

template<class morton_t, class _pos_t, u32 dim>
class irange_to_range;

namespace shamrock::sfc {

    template<class T>
    void details::sycl_fill_trailling_buffer(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<T>> &buf_morton) {

        shamlog_debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        if (fill_count - morton_count == 0) {
            shamlog_debug_sycl_ln(
                "MortonKernels", "sycl_fill_trailling_buffer skipping pow len 2 is ok");
            return;
        }

        sycl::range<1> range_npart{fill_count - morton_count};

        auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
            sycl::accessor m{*buf_morton, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<fill_trailling_buf<T>>(range_npart, [=](sycl::item<1> i) {
                m[morton_count + i.get_id()] = MortonInfo<T>::err_code;
            });
        };

        queue.submit(ker_fill_trailling_buf);
    }

    template void details::sycl_fill_trailling_buffer<u32>(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<u32>> &buf_morton);

    template void details::sycl_fill_trailling_buffer<u64>(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<u64>> &buf_morton);

    template<class morton_t, class _pos_t, u32 dim>
    void MortonKernels<morton_t, _pos_t, dim>::sycl_xyz_to_morton(
        sycl::queue &queue,
        u32 pos_count,
        sycl::buffer<pos_t> &in_positions,
        pos_t bounding_box_min,
        pos_t bounding_box_max,
        std::unique_ptr<sycl::buffer<morton_t>> &out_morton) {

        shamlog_debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        sycl::range<1> range_cnt{pos_count};

        queue.submit([&](sycl::handler &cgh) {
            auto transf = get_transform(bounding_box_min, bounding_box_max);

            sycl::accessor r{in_positions, cgh, sycl::read_only};
            sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<pos_to_morton<morton_t, pos_t, dim>>(
                range_cnt, [=](sycl::item<1> item) {
                    int i = (int) item.get_id(0);

                    ipos_t mr = to_morton_grid(r[i], transf);
                    m[i]      = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
                });
        }

        );
    }

    template<class morton_t, class _pos_t, u32 dim>
    void MortonKernels<morton_t, _pos_t, dim>::sycl_xyz_to_morton(
        sham::DeviceScheduler_ptr dev_sched,
        u32 pos_count,
        sham::DeviceBuffer<pos_t> &in_positions,
        pos_t bounding_box_min,
        pos_t bounding_box_max,
        std::unique_ptr<sycl::buffer<morton_t>> &out_morton) {

        shamlog_debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        sycl::range<1> range_cnt{pos_count};

        auto q = dev_sched->get_queue();

        sham::EventList el;
        auto r = in_positions.get_read_access(el);

        auto e = q.submit(el, [&](sycl::handler &cgh) {
            auto transf = get_transform(bounding_box_min, bounding_box_max);

            sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<pos_to_morton_usm<morton_t, pos_t, dim>>(
                range_cnt, [=](sycl::item<1> item) {
                    int i = (int) item.get_id(0);

                    ipos_t mr = to_morton_grid(r[i], transf);
                    m[i]      = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
                });
        });

        in_positions.complete_event_state(e);
    }

    template<class morton_t, class _pos_t, u32 dim>
    void MortonKernels<morton_t, _pos_t, dim>::sycl_irange_to_range(
        sycl::queue &queue,
        u32 buf_len,
        pos_t bounding_box_min,
        pos_t bounding_box_max,
        std::unique_ptr<sycl::buffer<ipos_t>> &buf_pos_min_cell,
        std::unique_ptr<sycl::buffer<ipos_t>> &buf_pos_max_cell,
        std::unique_ptr<sycl::buffer<pos_t>> &out_buf_pos_min_cell_flt,
        std::unique_ptr<sycl::buffer<pos_t>> &out_buf_pos_max_cell_flt) {
        sycl::range<1> range_cell{buf_len};

        constexpr u32 group_size = 256;
        u32 max_len              = buf_len;
        u32 group_cnt            = shambase::group_count(buf_len, group_size);
        group_cnt                = group_cnt + (group_cnt % 4);
        u32 corrected_len        = group_cnt * group_size;

        shamlog_debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        auto ker_convert_cell_ranges = [&, max_len](sycl::handler &cgh) {
            auto transf = get_transform(bounding_box_min, bounding_box_max);

            auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
            auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

            auto pos_min_cell_flt
                = sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
            auto pos_max_cell_flt
                = sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<irange_to_range<morton_t, pos_t, dim>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id      = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 gid           = group_tile_id * group_size + local_id;

                    if (gid >= max_len)
                        return;

                    pos_min_cell_flt[gid] = to_real_space(pos_min_cell[gid], transf);
                    pos_max_cell_flt[gid] = to_real_space(pos_max_cell[gid], transf);
                });
        };

        queue.submit(ker_convert_cell_ranges);
    }

    template class MortonKernels<u32, f32_3, 3>;
    template class MortonKernels<u64, f32_3, 3>;
    template class MortonKernels<u32, f64_3, 3>;
    template class MortonKernels<u64, f64_3, 3>;
    template class MortonKernels<u32, u32_3, 3>;
    template class MortonKernels<u64, u32_3, 3>;
    template class MortonKernels<u32, u64_3, 3>;
    template class MortonKernels<u64, u64_3, 3>;
    template class MortonKernels<u32, i64_3, 3>;
    template class MortonKernels<u64, i64_3, 3>;
} // namespace shamrock::sfc

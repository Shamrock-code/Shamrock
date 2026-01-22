// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file convert_ranges.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/integer.hpp"
#include "shamtree/kernels/convert_ranges.hpp"

template<>
void sycl_convert_cell_range<u32, f32_3>(
    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f32_3>> &buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f32_3>> &buf_pos_max_cell_flt) {

    using f3_xyzh = f32_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    constexpr u32 group_size = 256;
    u32 max_len              = leaf_cnt + internal_cnt;
    u32 group_cnt            = shambase::group_count(leaf_cnt + internal_cnt, group_size);
    group_cnt                = group_cnt + (group_cnt % 4);
    u32 corrected_len        = group_cnt * group_size;

    auto ker_convert_cell_ranges = [&, max_len](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt =
        // buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh); auto
        // pos_max_cell_flt =
        // buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor{
            *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor{
            *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u32_f32>(
            sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                u32 local_id      = id.get_local_id(0);
                u32 group_tile_id = id.get_group_linear_id();
                u32 gid           = group_tile_id * group_size + local_id;

                if (gid >= max_len)
                    return;

                pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0()) * (1 / 1024.f);
                pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0()) * (1 / 1024.f);

                pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1()) * (1 / 1024.f);
                pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1()) * (1 / 1024.f);

                pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2()) * (1 / 1024.f);
                pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2()) * (1 / 1024.f);

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;

                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;
            });
    };

    queue.submit(ker_convert_cell_ranges);
}

template<>
void sycl_convert_cell_range<u64, f32_3>(
    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f32_3>> &buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f32_3>> &buf_pos_max_cell_flt) {

    using f3_xyzh = f32_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    constexpr u32 group_size = 256;
    u32 max_len              = leaf_cnt + internal_cnt;
    u32 group_cnt            = shambase::group_count(leaf_cnt + internal_cnt, group_size);
    group_cnt                = group_cnt + (group_cnt % 4);
    u32 corrected_len        = group_cnt * group_size;

    auto ker_convert_cell_ranges = [&, max_len](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt =
        // buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh); auto
        // pos_max_cell_flt =
        // buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor{
            *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor{
            *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u64_f32>(
            sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                u32 local_id      = id.get_local_id(0);
                u32 group_tile_id = id.get_group_linear_id();
                u32 gid           = group_tile_id * group_size + local_id;

                if (gid >= max_len)
                    return;

                pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0()) * (1 / 2097152.f);
                pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0()) * (1 / 2097152.f);

                pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1()) * (1 / 2097152.f);
                pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1()) * (1 / 2097152.f);

                pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2()) * (1 / 2097152.f);
                pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2()) * (1 / 2097152.f);

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;

                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;
            });
    };

    queue.submit(ker_convert_cell_ranges);
}

template<>
void sycl_convert_cell_range<u32, f64_3>(
    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f64_3>> &buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f64_3>> &buf_pos_max_cell_flt) {

    using f3_xyzh = f64_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    constexpr u32 group_size = 256;
    u32 max_len              = leaf_cnt + internal_cnt;
    u32 group_cnt            = shambase::group_count(leaf_cnt + internal_cnt, group_size);
    group_cnt                = group_cnt + (group_cnt % 4);
    u32 corrected_len        = group_cnt * group_size;

    auto ker_convert_cell_ranges = [&, max_len](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt =
        // buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh); auto
        // pos_max_cell_flt =
        // buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor{
            *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor{
            *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u32_f64>(
            sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                u32 local_id      = id.get_local_id(0);
                u32 group_tile_id = id.get_group_linear_id();
                u32 gid           = group_tile_id * group_size + local_id;

                if (gid >= max_len)
                    return;

                pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0()) * (1 / 1024.);
                pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0()) * (1 / 1024.);

                pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1()) * (1 / 1024.);
                pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1()) * (1 / 1024.);

                pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2()) * (1 / 1024.);
                pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2()) * (1 / 1024.);

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;

                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;
            });
    };

    queue.submit(ker_convert_cell_ranges);
}

template<>
void sycl_convert_cell_range<u64, f64_3>(
    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f64_3>> &buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f64_3>> &buf_pos_max_cell_flt) {

    using f3_xyzh = f64_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    constexpr u32 group_size = 256;
    u32 max_len              = leaf_cnt + internal_cnt;
    u32 group_cnt            = shambase::group_count(leaf_cnt + internal_cnt, group_size);
    group_cnt                = group_cnt + (group_cnt % 4);
    u32 corrected_len        = group_cnt * group_size;

    auto ker_convert_cell_ranges = [&, max_len](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt =
        // buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh); auto
        // pos_max_cell_flt =
        // buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor{
            *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor{
            *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u64_f64>(
            sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                u32 local_id      = id.get_local_id(0);
                u32 group_tile_id = id.get_group_linear_id();
                u32 gid           = group_tile_id * group_size + local_id;

                if (gid >= max_len)
                    return;

                pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0()) * (1 / 2097152.);
                pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0()) * (1 / 2097152.);

                pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1()) * (1 / 2097152.);
                pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1()) * (1 / 2097152.);

                pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2()) * (1 / 2097152.);
                pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2()) * (1 / 2097152.);

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;

                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;
            });
    };

    queue.submit(ker_convert_cell_ranges);
}

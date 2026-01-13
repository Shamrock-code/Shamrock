// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file compute_ranges.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/integer.hpp"
#include "shambackends/math.hpp"
#include "shamtree/kernels/compute_ranges.hpp"

template<class u_morton>
void sycl_compute_cell_ranges(

    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> &buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>> &buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>> &buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> &buf_endrange,

    std::unique_ptr<sycl::buffer<typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr>>
        &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr>>
        &buf_pos_max_cell) {

    sycl::range<1> range_radix_tree{internal_cnt};

    constexpr u32 group_size = 256;
    u32 group_cnt            = shambase::group_count(internal_cnt, group_size);
    group_cnt                = group_cnt + (group_cnt % 4);
    u32 corrected_len        = group_cnt * group_size;

    auto ker_compute_cell_ranges = [&](sycl::handler &cgh) {
        auto morton_map    = buf_morton->template get_access<sycl::access::mode::read>(cgh);
        auto end_range_map = buf_endrange->get_access<sycl::access::mode::read>(cgh);

        auto pos_min_cell
            = buf_pos_min_cell->template get_access<sycl::access::mode::discard_write>(
                cgh); // was "write" before changed to fix warning
        auto pos_max_cell
            = buf_pos_max_cell->template get_access<sycl::access::mode::discard_write>(
                cgh); // was "write" before changed to fix warning

        auto rchild_flag = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto lchild_flag = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto rchild_id   = buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
        auto lchild_id   = buf_lchild_id->get_access<sycl::access::mode::read>(cgh);

        u32 internal_cell_cnt = internal_cnt;

        // Executing kernel
        cgh.parallel_for(sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
            u32 local_id      = id.get_local_id(0);
            u32 group_tile_id = id.get_group_linear_id();
            u32 gid           = group_tile_id * group_size + local_id;

            if (gid >= internal_cell_cnt)
                return;

            uint clz_ = sham::clz_xor(morton_map[gid], morton_map[end_range_map[gid]]);

            using Morton = shamrock::sfc::MortonCodes<u_morton, 3>;

            auto get_mask = [](u32 clz_) -> u_morton {
                if constexpr (std::is_same<u_morton, u64>::value) {
                    constexpr u64 mask_i = 0xFFFFFFFFFFFFFFFF;
                    return mask_i << (64U - clz_);
                }

                if constexpr (std::is_same<u_morton, u32>::value) {
                    constexpr u32 mask_i = 0xFFFFFFFF;
                    return mask_i << (32 - clz_);
                }
            };

            auto clz_offset   = Morton::get_offset(clz_);
            auto clz_offset_1 = Morton::get_offset(clz_ + 1);

            auto min_cell = Morton::morton_to_icoord(morton_map[gid] & get_mask(clz_));

            pos_min_cell[gid] = min_cell;

            pos_max_cell[gid] = clz_offset + min_cell;

            if (rchild_flag[gid]) {

                auto tmp = clz_offset - clz_offset_1;

                pos_min_cell[rchild_id[gid] + internal_cell_cnt] = min_cell + tmp;
                pos_max_cell[rchild_id[gid] + internal_cell_cnt] = clz_offset_1 + min_cell + tmp;
            }

            if (lchild_flag[gid]) {
                pos_min_cell[lchild_id[gid] + internal_cell_cnt] = min_cell;
                pos_max_cell[lchild_id[gid] + internal_cell_cnt] = clz_offset_1 + min_cell;
            }
        });
    };

    queue.submit(ker_compute_cell_ranges);
}

template void sycl_compute_cell_ranges(
    sycl::queue &queue,
    u32 leaf_cnt,
    u32 internal_cnt,
    std::unique_ptr<sycl::buffer<u32>> &buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> &buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>> &buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>> &buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> &buf_endrange,

    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>> &buf_pos_max_cell);

template void sycl_compute_cell_ranges(
    sycl::queue &queue,
    u32 leaf_cnt,
    u32 internal_cnt,
    std::unique_ptr<sycl::buffer<u64>> &buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> &buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>> &buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>> &buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> &buf_endrange,

    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>> &buf_pos_max_cell);

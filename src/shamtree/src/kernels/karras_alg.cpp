// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file karras_alg.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambackends/math.hpp"
#include "shamtree/kernels/karras_alg.hpp"
#include <stdexcept>

#define SGN(x) (x == 0) ? 0 : ((x > 0) ? 1 : -1)

template<class u_morton, class kername>
void __sycl_karras_alg(
    sycl::queue &queue,
    u32 internal_cell_count,
    sycl::buffer<u_morton> &in_morton,
    sycl::buffer<u32> &out_buf_lchild_id,
    sycl::buffer<u32> &out_buf_rchild_id,
    sycl::buffer<u8> &out_buf_lchild_flag,
    sycl::buffer<u8> &out_buf_rchild_flag,
    sycl::buffer<u32> &out_buf_endrange) {

    sycl::range<1> range_radix_tree{internal_cell_count};

    queue.submit([&](sycl::handler &cgh) {
        //@TODO add check if split count above 2G
        i32 morton_length = (i32) internal_cell_count + 1;

        auto m = in_morton.template get_access<sycl::access::mode::read>(cgh);

        auto lchild_id   = out_buf_lchild_id.get_access<sycl::access::mode::discard_write>(cgh);
        auto rchild_id   = out_buf_rchild_id.get_access<sycl::access::mode::discard_write>(cgh);
        auto lchild_flag = out_buf_lchild_flag.get_access<sycl::access::mode::discard_write>(cgh);
        auto rchild_flag = out_buf_rchild_flag.get_access<sycl::access::mode::discard_write>(cgh);
        auto end_range_cell = out_buf_endrange.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<kername>(range_radix_tree, [=](sycl::item<1> item) {
            int i = (int) item.get_id(0);

            auto DELTA = [=](i32 x, i32 y) {
                return sham::karras_delta(x, y, morton_length, m);
            };

            int ddelta = DELTA(i, i + 1) - DELTA(i, i - 1);

            int d = SGN(ddelta);

            // Compute upper bound for the length of the range
            int delta_min = DELTA(i, i - d);
            int lmax      = 2;
            while (DELTA(i, i + lmax * d) > delta_min) {
                lmax *= 2;
            }

            // Find the other end using
            int l = 0;
            int t = lmax / 2;
            while (t > 0) {
                if (DELTA(i, i + (l + t) * d) > delta_min) {
                    l = l + t;
                }
                t = t / 2;
            }
            int j = i + l * d;

            end_range_cell[i] = j;

            // Find the split position using binary search
            int delta_node = DELTA(i, j);
            int s          = 0;

            //@todo why float
            float div = 2;
            t         = sycl::ceil(l / div);
            while (true) {
                int tmp_ = i + (s + t) * d;
                if (DELTA(i, tmp_) > delta_node) {
                    s = s + t;
                }
                if (t <= 1)
                    break;
                div *= 2;
                t = sycl::ceil(l / div);
            }
            int gamma = i + s * d + sycl::min(d, 0);

            if (sycl::min(i, j) == gamma) {
                lchild_id[i]   = gamma;
                lchild_flag[i] = 1; // leaf
            } else {
                lchild_id[i]   = gamma;
                lchild_flag[i] = 0; // leaf
            }

            if (sycl::max(i, j) == gamma + 1) {
                rchild_id[i]   = gamma + 1;
                rchild_flag[i] = 1; // leaf
            } else {
                rchild_id[i]   = gamma + 1;
                rchild_flag[i] = 0; // leaf
            }
        });
    }

    );
}

class Kernel_Karras_alg_morton32;
class Kernel_Karras_alg_morton64;

template<>
void sycl_karras_alg<u32>(
    sycl::queue &queue,
    u32 internal_cell_count,
    sycl::buffer<u32> &in_morton,
    sycl::buffer<u32> &out_buf_lchild_id,
    sycl::buffer<u32> &out_buf_rchild_id,
    sycl::buffer<u8> &out_buf_lchild_flag,
    sycl::buffer<u8> &out_buf_rchild_flag,
    sycl::buffer<u32> &out_buf_endrange) {
    __sycl_karras_alg<u32, Kernel_Karras_alg_morton32>(
        queue,
        internal_cell_count,
        in_morton,
        out_buf_lchild_id,
        out_buf_rchild_id,
        out_buf_lchild_flag,
        out_buf_rchild_flag,
        out_buf_endrange);
}

template<>
void sycl_karras_alg<u64>(
    sycl::queue &queue,
    u32 internal_cell_count,
    sycl::buffer<u64> &in_morton,
    sycl::buffer<u32> &out_buf_lchild_id,
    sycl::buffer<u32> &out_buf_rchild_id,
    sycl::buffer<u8> &out_buf_lchild_flag,
    sycl::buffer<u8> &out_buf_rchild_flag,
    sycl::buffer<u32> &out_buf_endrange) {
    __sycl_karras_alg<u64, Kernel_Karras_alg_morton64>(
        queue,
        internal_cell_count,
        in_morton,
        out_buf_lchild_id,
        out_buf_rchild_id,
        out_buf_lchild_flag,
        out_buf_rchild_flag,
        out_buf_endrange);
}

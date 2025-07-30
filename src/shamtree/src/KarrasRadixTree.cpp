// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file KarrasRadixTree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include <utility>

/// Macro to get the sign of a number
#define SGN(x) (x == 0) ? 0 : ((x > 0) ? 1 : -1)

/**
 * @brief Karras 2012 algorithm with addition endrange buffer
 *
 * Given a list of morton codes, compute the left and right child id, left and right
 * child flag, and the endrange for each cell using the Karras 2012 algorithm.
 *
 * @param[in] queue sycl queue
 * @param[in] internal_cell_count number of internal cells
 * @param[in] in_morton input morton codes
 * @param[out] out_buf_lchild_id left child id
 * @param[out] out_buf_rchild_id right child id
 * @param[out] out_buf_lchild_flag left child flag
 * @param[out] out_buf_rchild_flag right child flag
 * @param[out] out_buf_endrange endrange
 */
template<class u_morton>
void __karras_alg(
    sham::DeviceQueue &queue,
    u32 internal_cell_count,
    sham::DeviceBuffer<u_morton> &in_morton,
    sham::DeviceBuffer<u32> &out_buf_lchild_id,
    sham::DeviceBuffer<u32> &out_buf_rchild_id,
    sham::DeviceBuffer<u8> &out_buf_lchild_flag,
    sham::DeviceBuffer<u8> &out_buf_rchild_flag,
    sham::DeviceBuffer<u32> &out_buf_endrange) {

    // Early return if the tree is a single leaf as there is no tree structure in this case.
    if (internal_cell_count == 0) {
        return;
    }

    sham::kernel_call(
        queue,
        sham::MultiRef{in_morton},
        sham::MultiRef{
            out_buf_lchild_id,
            out_buf_rchild_id,
            out_buf_lchild_flag,
            out_buf_rchild_flag,
            out_buf_endrange},
        internal_cell_count,

        [morton_length = internal_cell_count + 1](
            u32 i,
            const u_morton *__restrict morton,
            u32 *__restrict lchild_id,
            u32 *__restrict rchild_id,
            u8 *__restrict lchild_flag,
            u8 *__restrict rchild_flag,
            u32 *__restrict end_range_cell) {
            auto DELTA = [=](i32 x, i32 y) {
                return sham::karras_delta(x, y, morton_length, morton);
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
            u32 j = i + l * d;

            end_range_cell[i] = j;

            // Find the split position using binary search
            int delta_node = DELTA(i, j);
            int s          = 0;

            // TODO why float
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

namespace shamtree {

    template<class Tmorton>
    inline u32 get_tree_depth() {
        return shambase::bitsizeof<Tmorton>;
    }

    template<class Tmorton>
    KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes,
        KarrasRadixTree &&recycled_tree) {

        u32 internal_cell_count = morton_count - 1;

        recycled_tree.tree_depth = get_tree_depth<Tmorton>();

        recycled_tree.buf_lchild_id.resize(internal_cell_count);
        recycled_tree.buf_rchild_id.resize(internal_cell_count);
        recycled_tree.buf_lchild_flag.resize(internal_cell_count);
        recycled_tree.buf_rchild_flag.resize(internal_cell_count);
        recycled_tree.buf_endrange.resize(internal_cell_count);

        __karras_alg(
            dev_sched->get_queue(),
            internal_cell_count,
            morton_codes,
            recycled_tree.buf_lchild_id,
            recycled_tree.buf_rchild_id,
            recycled_tree.buf_lchild_flag,
            recycled_tree.buf_rchild_flag,
            recycled_tree.buf_endrange);

        return std::forward<KarrasRadixTree>(recycled_tree);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Shortcut without caching
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Tmorton>
    KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes) {

        u32 internal_cell_count = morton_count - 1;

        sham::DeviceBuffer<u32> buf_lchild_id(internal_cell_count, dev_sched);
        sham::DeviceBuffer<u32> buf_rchild_id(internal_cell_count, dev_sched);
        sham::DeviceBuffer<u8> buf_lchild_flag(internal_cell_count, dev_sched);
        sham::DeviceBuffer<u8> buf_rchild_flag(internal_cell_count, dev_sched);
        sham::DeviceBuffer<u32> buf_endrange(internal_cell_count, dev_sched);

        KarrasRadixTree tree(
            std::move(buf_lchild_id),
            std::move(buf_rchild_id),
            std::move(buf_lchild_flag),
            std::move(buf_rchild_flag),
            std::move(buf_endrange),
            get_tree_depth<Tmorton>());

        return karras_tree_from_morton_set(dev_sched, morton_count, morton_codes, std::move(tree));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Explicit instantiations
    ////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN // To avoid doxygen complaining like always ...
    template KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<u32> &morton_codes,
        KarrasRadixTree &&recycled_tree);

    template KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<u32> &morton_codes);

    template KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<u64> &morton_codes,
        KarrasRadixTree &&recycled_tree);

    template KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<u64> &morton_codes);
#endif

    std::string karras_tree_to_dot_graph(KarrasRadixTree &tree) {

        std::vector<u32> lchild_id  = {};
        std::vector<u8> lchild_flag = {};
        std::vector<u32> rchild_id  = {};
        std::vector<u8> rchild_flag = {};
        std::vector<u32> endrange   = {};

        lchild_id   = tree.buf_lchild_id.copy_to_stdvec();
        rchild_id   = tree.buf_rchild_id.copy_to_stdvec();
        lchild_flag = tree.buf_lchild_flag.copy_to_stdvec();
        rchild_flag = tree.buf_rchild_flag.copy_to_stdvec();
        endrange    = tree.buf_endrange.copy_to_stdvec();

        std::string dot_graph = "";

        dot_graph += "digraph G {\n";
        dot_graph += "rankdir=LR;\n";

        for (u32 i = 0; i < tree.get_internal_cell_count(); ++i) {

            if (lchild_flag[i] == 0) {
                dot_graph
                    += "i" + std::to_string(i) + " -> i" + std::to_string(lchild_id[i]) + ";\n";
            } else {
                dot_graph
                    += "i" + std::to_string(i) + " -> l" + std::to_string(lchild_id[i]) + ";\n";
            }

            if (rchild_flag[i] == 0) {
                dot_graph
                    += "i" + std::to_string(i) + " -> i" + std::to_string(rchild_id[i]) + ";\n";
            } else {
                dot_graph
                    += "i" + std::to_string(i) + " -> l" + std::to_string(rchild_id[i]) + ";\n";
            }
        }

        dot_graph += "}\n";
        return dot_graph;
    }

} // namespace shamtree

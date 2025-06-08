// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file KarrasRadixTreeAABB.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include <functional>
#include <utility>

namespace shamtree {
    /**
     * @class KarrasRadixTree
     * @brief A data structure representing a Karras Radix Tree.
     *
     * This class encapsulates the structure of a Karras Radix Tree, which is used for efficiently
     * handling hierarchical data based on Morton codes. It manages buffers for left and right child
     * identifiers and flags, as well as end ranges.
     */
    template<class Tvec>
    class KarrasRadixTreeAABB;
} // namespace shamtree

template<class Tvec>
class shamtree::KarrasRadixTreeAABB {

    public:
    /// Get internal cell count
    inline u32 get_total_cell_count() { return buf_cell_min.get_size(); }

    sham::DeviceBuffer<Tvec> buf_cell_min; ///< left child id (size = internal_count)
    sham::DeviceBuffer<Tvec> buf_cell_max; ///< right child id (size = internal_count)

    /// CTOR
    KarrasRadixTreeAABB(
        sham::DeviceBuffer<Tvec> &&buf_cell_min, sham::DeviceBuffer<Tvec> &&buf_cell_max)
        : buf_cell_min(std::move(buf_cell_min)), buf_cell_max(std::move(buf_cell_max)) {}
};

namespace shamtree {

    template<class Tvec>
    inline KarrasRadixTreeAABB<Tvec> prepare_karras_radix_tree_aabb(
        const KarrasRadixTree &tree, KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb) {

        KarrasRadixTreeAABB<Tvec> ret = std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb);

        ret.buf_cell_min.resize(tree.get_total_cell_count());
        ret.buf_cell_max.resize(tree.get_total_cell_count());

        return ret;
    }

    template<class Tvec>
    inline KarrasRadixTreeAABB<Tvec>
    propagate_aabb_up(KarrasRadixTreeAABB<Tvec> &tree_aabb, const KarrasRadixTree &tree) {

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        auto step = [&]() {
            auto traverser = tree.get_structure_traverser();

            sham::kernel_call(
                q,
                sham::MultiRef{traverser},
                sham::MultiRef{tree_aabb.buf_cell_min, tree_aabb.buf_cell_max},
                tree.get_internal_cell_count(),
                [=](u32 gid,
                    auto tree_traverser,
                    Tvec *__restrict cell_min,
                    Tvec *__restrict cell_max) {
                    u32 left_child  = tree_traverser.get_left_child(gid);
                    u32 right_child = tree_traverser.get_right_child(gid);

                    Tvec bminl = cell_min[left_child];
                    Tvec bminr = cell_min[right_child];
                    Tvec bmaxl = cell_max[left_child];
                    Tvec bmaxr = cell_max[right_child];

                    Tvec bmin = sham::min(bminl, bminr);
                    Tvec bmax = sham::max(bmaxl, bmaxr);

                    cell_min[gid] = bmin;
                    cell_max[gid] = bmax;
                });
        };

        for (u32 i = 0; i < tree.tree_depth; i++) {
            step();
        }
    }

    /**
     * @brief Compute the AABB of all cells in the tree.
     *
     * @param tree The tree to compute the AABBs for.
     * @param iter The cell iterator to use to compute the AABBs.
     * @param recycled_tree_aabb The tree AABBs to recycle.
     * @param fct_fill_leaf The function to use to compute the AABBs of the leaf cells.
     *
     * @return The tree AABBs.
     */
    template<class Tvec>
    inline KarrasRadixTreeAABB<Tvec> compute_tree_aabb(
        const KarrasRadixTree &tree,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        std::function<void(KarrasRadixTreeAABB<Tvec> &, u32)> fct_fill_leaf) {

        auto aabbs = prepare_karras_radix_tree_aabb(
            tree, std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb));

        fct_fill_leaf(aabbs, tree.get_internal_cell_count());

        propagate_aabb_up(aabbs, tree);
    }

} // namespace shamtree

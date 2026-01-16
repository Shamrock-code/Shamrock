// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file KarrasRadixTreeAABB.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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
#include "shamtree/LeafCellIterator.hpp"
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
    inline u32 get_total_cell_count() { return buf_aabb_min.get_size(); }

    sham::DeviceBuffer<Tvec> buf_aabb_min; ///< left child id (size = internal_count)
    sham::DeviceBuffer<Tvec> buf_aabb_max; ///< right child id (size = internal_count)

    /// CTOR
    KarrasRadixTreeAABB(
        sham::DeviceBuffer<Tvec> &&buf_cell_min, sham::DeviceBuffer<Tvec> &&buf_cell_max)
        : buf_aabb_min(std::move(buf_cell_min)), buf_aabb_max(std::move(buf_cell_max)) {}

    static inline KarrasRadixTreeAABB make_empty(sham::DeviceScheduler_ptr dev_sched) {
        return KarrasRadixTreeAABB{
            sham::DeviceBuffer<Tvec>(0, dev_sched), sham::DeviceBuffer<Tvec>(0, dev_sched)};
    }
};

namespace shamtree {

    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> new_empty_karras_radix_tree_aabb() {
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        return KarrasRadixTreeAABB<Tvec>(
            sham::DeviceBuffer<Tvec>(0, dev_sched), sham::DeviceBuffer<Tvec>(0, dev_sched));
    }

    /**
     * @brief Prepare a KarrasRadixTreeAABB from a KarrasRadixTree.
     *
     * This function prepares a KarrasRadixTreeAABB from a KarrasRadixTree. It allocates the
     * necessary buffers to store the AABBs of all cells in the tree and recycles the buffers
     * from the recycled_tree_aabb argument if possible.
     *
     * @param tree The KarrasRadixTree to prepare the AABBs for.
     * @param recycled_tree_aabb The KarrasRadixTreeAABB to recycle the buffers from.
     *
     * @return The prepared KarrasRadixTreeAABB.
     */
    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> prepare_karras_radix_tree_aabb(
        const KarrasRadixTree &tree, KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb);

    /**
     * @brief Propagates the axis-aligned bounding boxes (AABBs) upwards in the tree.
     *
     * This function updates the AABBs for internal nodes in a KarrasRadixTree by
     * combining the AABBs of their child nodes. It iteratively traverses the nodes
     * of the tree, computing the minimum and maximum bounds for each internal node
     * by taking the minimum and maximum of the bounds of its left and right children.
     *
     * @tparam Tvec The vector type used for the AABB bounds.
     * @param tree_aabb A reference to the KarrasRadixTreeAABB containing the AABBs
     *                  of the tree nodes.
     * @param tree The KarrasRadixTree whose structure is used to determine the
     *             parent-child relationships.
     */
    template<class Tvec>
    void propagate_aabb_up(KarrasRadixTreeAABB<Tvec> &tree_aabb, const KarrasRadixTree &tree);

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
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb(
        const KarrasRadixTree &tree,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        const std::function<void(KarrasRadixTreeAABB<Tvec> &, u32)> &fct_fill_leaf);

    /**
     * @brief Compute the AABB of all cells in the tree from positions.
     *
     * This function computes the AABBs of all cells in the tree by iterating over the
     * objects in each cell and computing the minimum and maximum bounds.
     *
     * @param tree The tree to compute the AABBs for.
     * @param cell_it The cell iterator to use to access the cells.
     * @param recycled_tree_aabb The tree AABBs to recycle.
     * @param positions The buffer of positions to use for computing the AABBs.
     *
     * @return The tree AABBs.
     */
    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb_from_positions(
        const KarrasRadixTree &tree,
        const LeafCellIterator &cell_it,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        sham::DeviceBuffer<Tvec> &positions);

    /// same but for position ranges
    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb_from_position_ranges(
        const KarrasRadixTree &tree,
        const LeafCellIterator &cell_it,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        sham::DeviceBuffer<Tvec> &min,
        sham::DeviceBuffer<Tvec> &max);

} // namespace shamtree

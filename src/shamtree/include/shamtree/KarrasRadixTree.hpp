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
 * @file KarrasRadixTree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shamtree/KarrasTreeTraverser.hpp"

namespace shamtree {

    /**
     * @class KarrasRadixTree
     * @brief A data structure representing a Karras Radix Tree.
     *
     * This class encapsulates the structure of a Karras Radix Tree, which is used for efficiently
     * handling hierarchical data based on Morton codes. It manages buffers for left and right child
     * identifiers and flags, as well as end ranges.
     */
    class KarrasRadixTree;

} // namespace shamtree

class shamtree::KarrasRadixTree {

    public:
    /// Get internal cell count
    inline u32 get_internal_cell_count() const { return buf_lchild_id.get_size(); }

    /// Get leaf count
    inline u32 get_leaf_count() const { return get_internal_cell_count() + 1; }

    inline u32 get_total_cell_count() const { return get_internal_cell_count() + get_leaf_count(); }

    sham::DeviceBuffer<u32> buf_lchild_id;  ///< left child id (size = internal_count)
    sham::DeviceBuffer<u32> buf_rchild_id;  ///< right child id (size = internal_count)
    sham::DeviceBuffer<u8> buf_lchild_flag; ///< left child flag (size = internal_count)
    sham::DeviceBuffer<u8> buf_rchild_flag; ///< right child flag (size = internal_count)
    sham::DeviceBuffer<u32> buf_endrange;   ///< endrange (size = internal_count)

    u32 tree_depth;

    /// CTOR
    KarrasRadixTree(
        sham::DeviceBuffer<u32> &&buf_lchild_id,
        sham::DeviceBuffer<u32> &&buf_rchild_id,
        sham::DeviceBuffer<u8> &&buf_lchild_flag,
        sham::DeviceBuffer<u8> &&buf_rchild_flag,
        sham::DeviceBuffer<u32> &&buf_endrange,
        u32 tree_depth)
        : buf_lchild_id(std::move(buf_lchild_id)), buf_rchild_id(std::move(buf_rchild_id)),
          buf_lchild_flag(std::move(buf_lchild_flag)), buf_rchild_flag(std::move(buf_rchild_flag)),
          buf_endrange(std::move(buf_endrange)), tree_depth(tree_depth) {}

    inline KarrasTreeTraverser get_structure_traverser() const {
        return KarrasTreeTraverser{
            buf_lchild_id,
            buf_rchild_id,
            buf_lchild_flag,
            buf_rchild_flag,
            get_internal_cell_count()};
    }
};

namespace shamtree {

    /**
     * @brief Constructs a KarrasRadixTree from a set of reduced Morton codes.
     *
     * This function builds a KarrasRadixTree using the provided set of Morton codes.
     * The tree is constructed by resizing the buffers for left and right child IDs,
     * child flags, and end ranges to accommodate the internal cells, then executing
     * the Karras algorithm on the device scheduler's queue.
     *
     * @tparam Tmorton The type of the Morton codes.
     * @param[in] dev_sched A pointer to the device scheduler used for managing
     *                      execution on the device.
     * @param[in] morton_count The number of Morton codes provided.
     * @param[in] morton_codes A device buffer containing the Morton codes.
     * @param[in,out] recycled_tree A KarrasRadixTree object from which we will reuse the allocs.
     * @return A KarrasRadixTree object populated with the tree structure based
     *         on the provided Morton codes.
     */
    template<class Tmorton>
    KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes,
        KarrasRadixTree &&recycled_tree);

    /**
     * @brief Constructs a KarrasRadixTree from a set of reduced Morton codes without reuse.
     *
     * Equivalent to karras_tree_from_morton_set but performs the allocations.
     *
     * @tparam Tmorton The type of the Morton codes.
     * @param[in] dev_sched A pointer to the device scheduler used for managing
     *                      execution on the device.
     * @param[in] morton_count The number of Morton codes provided.
     * @param[in] morton_codes A device buffer containing the Morton codes.
     * @return A KarrasRadixTree object populated with the tree structure based
     *         on the provided Morton codes.
     */
    template<class Tmorton>
    KarrasRadixTree karras_tree_from_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes);

    /// Get tree as dot graph
    std::string karras_tree_to_dot_graph(KarrasRadixTree &recycled_tree);

} // namespace shamtree

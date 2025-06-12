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
 * @file CLBVHObjectIterator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/KarrasRadixTreeAABB.hpp"
#include "shamtree/MortonReducedSet.hpp"

namespace shamtree {

    /// Accessed version of CLBVHObjectIterator
    template<class Tmorton, class Tvec, u32 dim>
    struct CLBVHObjectIteratorAccessed;

    /**
     * @class CLBVHObjectIterator
     * @brief
     * This class is designed to traverse a BVH tree represented as a
     * Compressed Leaf BVH (CLBVH) and a Karras Radix Tree.
     *
     * @tparam Tmorton type of the morton codes
     * @tparam Tvec type of the vector (usually a float_3)
     * @tparam dim dimensionality of the problem
     */
    template<class Tmorton, class Tvec, u32 dim>
    struct CLBVHObjectIterator;

} // namespace shamtree

template<class Tmorton, class Tvec, u32 dim>
struct shamtree::CLBVHObjectIteratorAccessed {

    /// maximum depth of the tree according to the morton codes
    static constexpr u32 tree_depth_max
        = shamrock::sfc::MortonCodes<Tmorton, 3>::significant_bits + 1;

    CellIterator::acc cell_iterator;            ///< Cell iterator
    KarrasTreeTraverserAccessed tree_traverser; ///< Tree traverser
    const Tvec *aabb_min;                       ///< Minimum of the AABB
    const Tvec *aabb_max;                       ///< Maximum of the AABB

    /**
     * @brief Traverses the tree by calling tree_traverser's stack_based_traversal.
     *
     * This function is a convenience wrapper around the stack_based_traversal
     * function of the tree_traverser.
     *
     * This function is a shorthand to supply the tree depth
     *
     * @param[in] traverse_condition a function taking a node_id and returning a
     * boolean indicating whether to traverse the node further or not.
     * @param[in] on_found_leaf a function taking a node_id and being called when a
     * leaf node is reached.
     * @param[in] on_excluded_node a function taking a node_id and being called when
     * a node is excluded from the traversal.
     */
    template<class Functor1, class Functor2, class Functor3>
    inline void traverse_tree_base(
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        tree_traverser.template stack_based_traversal<tree_depth_max>(
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }

    /**
     * @brief Traverses the tree and executes a function for each found object.
     *
     * @param[in] traverse_condition_with_aabb A function taking a node_id and its AABB,
     * and returning a boolean indicating whether to traverse the node further.
     * @param[in] on_found_object A function to be called for each object found in a leaf
     * node that meets the traversal condition.
     */
    template<class Functor1, class Functor2>
    inline void
    rtree_for(Functor1 &&traverse_condition_with_aabb, Functor2 &&on_found_object) const {

        traverse_tree_base(
            [&](u32 node_id) { // interaction crit
                return traverse_condition_with_aabb(
                    node_id, shammath::AABB<Tvec>{aabb_min[node_id], aabb_max[node_id]});
            },
            [&](u32 node_id) { // on object found
                u32 leaf_id = node_id - tree_traverser.offset_leaf;
                cell_iterator.for_each_in_cell(leaf_id, on_found_object);
            },
            [&](u32) {});
    }
};

template<class Tmorton, class Tvec, u32 dim>
struct shamtree::CLBVHObjectIterator {
    CellIterator cell_iterator;               ///< Cell iterator
    KarrasTreeTraverser tree_traverser;       ///< Tree traverser
    const sham::DeviceBuffer<Tvec> &aabb_min; ///< Minimum of the AABB
    const sham::DeviceBuffer<Tvec> &aabb_max; ///< Maximum of the AABB

    /// shorthand for CLBVHObjectIteratorAccessed
    using acc = CLBVHObjectIteratorAccessed<Tmorton, Tvec, dim>;

    /// get read only accessor
    inline acc get_read_access(sham::EventList &deps) const {
        return acc{
            cell_iterator.get_read_access(deps),
            tree_traverser.get_read_access(deps),
            aabb_min.get_read_access(deps),
            aabb_max.get_read_access(deps)};
    }

    /// complete the buffer states with the resulting event
    inline void complete_event_state(sycl::event e) const {
        cell_iterator.complete_event_state(e);
        tree_traverser.complete_event_state(e);
        aabb_min.complete_event_state(e);
        aabb_max.complete_event_state(e);
    }
};

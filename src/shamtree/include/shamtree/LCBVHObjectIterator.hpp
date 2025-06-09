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
 * @file LCBVHObjectIterator.hpp
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

    template<class Tmorton, class Tvec, u32 dim>
    struct LCBVHObjectIteratorAccessed;

    template<class Tmorton, class Tvec, u32 dim>
    struct LCBVHObjectIterator;

} // namespace shamtree

template<class Tmorton, class Tvec, u32 dim>
struct shamtree::LCBVHObjectIteratorAccessed {

    static constexpr u32 tree_depth_max
        = shamrock::sfc::MortonCodes<Tmorton, 3>::significant_bits + 1;

    CellIterator::acc cell_iterator;
    KarrasTreeTraverserAccessed tree_traverser;
    const Tvec *aabb_min;
    const Tvec *aabb_max;

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

    template<class Functor1, class Functor2>
    inline void
    rtree_for(Functor1 &&traverse_condition_with_aabb, Functor2 &&on_found_object) const {

        traverse_tree_base(
            [&](u32 node_id) {
                return traverse_condition_with_aabb(
                    node_id, shammath::AABB<Tvec>{aabb_min[node_id], aabb_max[node_id]});
            },
            [&](u32 node_id) {
                u32 leaf_id = node_id - tree_traverser.offset_leaf;
                cell_iterator.for_each_in_cell(leaf_id, on_found_object);
            },
            [&](u32) {});
    }
};

template<class Tmorton, class Tvec, u32 dim>
struct shamtree::LCBVHObjectIterator {
    CellIterator cell_iterator;
    KarrasTreeTraverser tree_traverser;
    const sham::DeviceBuffer<Tvec> &aabb_min;
    const sham::DeviceBuffer<Tvec> &aabb_max;

    using acc = LCBVHObjectIteratorAccessed<Tmorton, Tvec, dim>;

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

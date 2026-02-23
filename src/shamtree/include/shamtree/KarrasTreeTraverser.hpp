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
 * @file KarrasTreeTraverser.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <vector>

namespace shamtree {

    /**
     * @struct KarrasTreeTraverser
     * @brief Utility struct to traverse a Karras Radix Tree
     */
    struct KarrasTreeTraverser;

    /// host version of the traverser
    struct KarrasTreeTraverserHost;

    /// read only accessor to buffer data
    struct KarrasTreeTraverserAccessed;

} // namespace shamtree

struct shamtree::KarrasTreeTraverserAccessed {
    const u32 *lchild_id;
    const u32 *rchild_id;
    const u8 *lchild_flag;
    const u8 *rchild_flag;
    u32 offset_leaf;

    /**
     * @brief Retrieves the left child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the left child.
     * @return The ID of the left child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_left_child(u32 id) const {
        return lchild_id[id] + offset_leaf * u32(lchild_flag[id]);
    }

    /**
     * @brief Retrieves the right child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the right child.
     * @return The ID of the right child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_right_child(u32 id) const {
        return rchild_id[id] + offset_leaf * u32(rchild_flag[id]);
    }

    /// is the given id a leaf (Note that if there is no internal cell every node is a leaf)
    inline bool is_id_leaf(u32 id) const { return id >= offset_leaf; }

    /// stack based tree traversal
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        u32 root_node,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        static constexpr u32 _nindex = 4294967295;

        // Init the stack state
        std::array<u32, tree_depth> id_stack;

        u32 stack_cursor       = tree_depth - 1;
        id_stack[stack_cursor] = root_node;

        // until the stack is empty
        while (stack_cursor < tree_depth) {

            // Pop the top of the stack
            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = _nindex;
            stack_cursor++;

            // check iteraction creteria
            bool cur_id_valid = traverse_condition(current_node_id);

            if (cur_id_valid) { // leaf or cell satisfies the criteria

                if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                    on_found_leaf(current_node_id);

                } else { // it can interact & not leaf => stack

                    u32 lid = get_left_child(current_node_id);
                    u32 rid = get_right_child(current_node_id);

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // This does not satisfy the criteria => excluded case (gravity for ex.)
                on_excluded_node(current_node_id);
            }
        }
    }

    /// stack based tree traversal
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        stack_based_traversal<tree_depth>(
            root_node,
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }
};

struct shamtree::KarrasTreeTraverser {

    const sham::DeviceBuffer<u32> &buf_lchild_id;  ///< ref to left child id buffer
    const sham::DeviceBuffer<u32> &buf_rchild_id;  ///< ref to right child id buffer
    const sham::DeviceBuffer<u8> &buf_lchild_flag; ///< ref to left child flag buffer
    const sham::DeviceBuffer<u8> &buf_rchild_flag; ///< ref to right child flag buffer
    u32 offset_leaf; ///< how many internal nodes before the first leaf ?

    /// get read only accessor
    inline KarrasTreeTraverserAccessed get_read_access(sham::EventList &deps) const {
        return KarrasTreeTraverserAccessed{
            buf_lchild_id.get_read_access(deps),
            buf_rchild_id.get_read_access(deps),
            buf_lchild_flag.get_read_access(deps),
            buf_rchild_flag.get_read_access(deps),
            offset_leaf};
    }

    /// complete the buffer states with the resulting event
    inline void complete_event_state(sycl::event e) const {
        buf_lchild_id.complete_event_state(e);
        buf_rchild_id.complete_event_state(e);
        buf_lchild_flag.complete_event_state(e);
        buf_rchild_flag.complete_event_state(e);
    }

    /// is the root a leaf ?
    inline bool is_root_leaf() const { return offset_leaf == 0; }
};

struct shamtree::KarrasTreeTraverserHost {

    std::vector<u32> buf_lchild_id;  ///< ref to left child id buffer
    std::vector<u32> buf_rchild_id;  ///< ref to right child id buffer
    std::vector<u8> buf_lchild_flag; ///< ref to left child flag buffer
    std::vector<u8> buf_rchild_flag; ///< ref to right child flag buffer
    u32 offset_leaf;                 ///< how many internal nodes before the first leaf ?

    /// get read only accessor
    inline KarrasTreeTraverserAccessed get_read_access() const {
        return KarrasTreeTraverserAccessed{
            buf_lchild_id.data(),
            buf_rchild_id.data(),
            buf_lchild_flag.data(),
            buf_rchild_flag.data(),
            offset_leaf};
    }
};

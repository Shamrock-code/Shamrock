// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SerialPatchTree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/log.hpp"

/*
 * This file implements a faster version of the following algorithm (commit 3ff163a25), which
 * builds the serial tree with a serial iterative pre-order traversal, run once per root, where each
 * node is converted as soon as it is visited and its id written back in the childs_id of its
 * parent.
 *
 *     struct TraversalItem {
 *         u64 key;        // key of the node in ptree.tree
 *         u64 parent_id;  // id of its parent in the serial tree (u64_max for a root)
 *         u32 child_slot; // which child of the parent it is
 *     };
 *
 *     std::vector<TraversalItem> stack{{root_key, u64_max, 0}};
 *     while (!stack.empty()) {
 *         TraversalItem cur = stack.back();
 *         stack.pop_back();
 *
 *         const PatchTree::Node &ptn = ptree.tree.at(cur.key);
 *
 *         u64 id = counter++;
 *         if (cur.parent_id != u64_max) {
 *             serial_tree[cur.parent_id].childs_id[cur.child_slot] = id;
 *         }
 *
 *         max_level = sycl::max(ptn.tree_node.level, max_level);
 *         serial_tree.push_back(ptn.convert(box_transform));
 *         linked_patch_ids.push_back(ptn.linked_patchid);
 *
 *         if (!ptn.tree_node.is_leaf) {
 *             for (u32 i = 8; i-- > 0;) { // reversed so that child 0 is visited first
 *                 stack.push_back({ptn.tree_node.childs_nid[i], id, i});
 *             }
 *         }
 *     }
 *
 * The two pass version implemented here is equivalent up to the numbering of the nodes : it
 * produces the same nodes (boxes, linked patch ids, leafs), the same childs in the same order, the
 * same roots in the same order and the same level count, but numbers the nodes in breadth-first
 * order instead of pre-order. Every user of the serial tree goes through roots_ids and childs_id,
 * so the numbering does not matter. The test "shamrock/scheduler/SerialPatchTree:build" checks this
 * equivalence on random patch trees.
 */

namespace {

    using PatchTree = shamrock::scheduler::PatchTree;
    using Node      = PatchTree::Node;

    // Below these sizes the OpenMP startup cost outweighs the gain (SerialPatchTree benchmark:
    // serial is faster at ~40k nodes, parallel is ~3x faster at ~300k nodes)
    constexpr u64 omp_min_lookups = 1U << 16;
    constexpr u64 omp_min_nodes   = 1U << 17;

    /// Order of the patch tree nodes in the serial tree
    struct SerialTreeLayout {
        /// nodes[i] is the patch tree node that becomes node i of the serial tree
        std::vector<const Node *> nodes;
        /// the childs of node i are first_child_ids[i] + 0..7 (u64_max for a leaf)
        std::vector<u64> first_child_ids;
        /// ids of the roots, in the order of PatchTree::roots_id
        std::vector<u64> roots_ids;
    };

    /**
     * @brief List the patch tree nodes in breadth-first order, roots first
     *
     * The 8 childs of a node are listed next to each other, so their ids are known as soon as
     * their parent is listed. The lookups of a level are independent of each other, hence they
     * are done in parallel.
     */
    SerialTreeLayout make_breadth_first_layout(const PatchTree &ptree) {
        SerialTreeLayout layout;
        auto &nodes           = layout.nodes;
        auto &first_child_ids = layout.first_child_ids;

        // is_leaf[i] mirrors nodes[i]->tree_node.is_leaf, filled when the node is looked up, so
        // that the serial loop below reads a compact array instead of following the pointers.
        // (u8 rather than bool since std::vector<bool> can not be written concurrently)
        std::vector<u8> is_leaf;

        nodes.reserve(ptree.tree.size());
        first_child_ids.reserve(ptree.tree.size());
        is_leaf.reserve(ptree.tree.size());

        for (u64 root_key : ptree.roots_id) {
            const Node &root = ptree.tree.at(root_key);
            layout.roots_ids.push_back(nodes.size());
            nodes.push_back(&root);
            is_leaf.push_back(root.tree_node.is_leaf);
        }

        // nodes[level_begin, level_end) is the current level
        u64 level_begin = 0;
        u64 level_end   = nodes.size();
        while (level_begin < level_end) {

            // give their ids to the childs of the current level
            u64 next_level_end = level_end;
            for (u64 i = level_begin; i < level_end; i++) {
                if (is_leaf[i]) {
                    first_child_ids.push_back(u64_max);
                } else {
                    first_child_ids.push_back(next_level_end);
                    next_level_end += 8;
                }
            }
            nodes.resize(next_level_end);
            is_leaf.resize(next_level_end);

            // look the childs up in the patch tree
            bool missing_child = false;
#pragma omp parallel for if (next_level_end - level_end > omp_min_lookups)                         \
    reduction(|| : missing_child)
            for (u64 i = level_begin; i < level_end; i++) {
                if (first_child_ids[i] == u64_max) {
                    continue;
                }
                for (u32 c = 0; c < 8; c++) {
                    auto it = ptree.tree.find(nodes[i]->tree_node.childs_nid[c]);
                    if (it == ptree.tree.end()) {
                        missing_child = true;
                        continue;
                    }
                    nodes[first_child_ids[i] + c]   = &it->second;
                    is_leaf[first_child_ids[i] + c] = it->second.tree_node.is_leaf;
                }
            }
            if (missing_child) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "a child referenced in the patch tree is not in the tree");
            }

            level_begin = level_end;
            level_end   = next_level_end;
        }

        return layout;
    }

} // namespace

template<class vec>
void SerialPatchTree<vec>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<vec> box_transform) {

    // pass 1 : choose where each node goes in the serial tree
    SerialTreeLayout layout = make_breadth_first_layout(ptree);

    roots_ids  = std::move(layout.roots_ids);
    root_count = roots_ids.size();

    // pass 2 : convert the nodes, in parallel since this writes the whole serial tree
    u64 node_count = layout.nodes.size();
    serial_tree.resize(node_count);
    linked_patch_ids.resize(node_count);

    u32 max_level = 0;
#pragma omp parallel for if (node_count > omp_min_nodes) reduction(max : max_level)
    for (u64 i = 0; i < node_count; i++) {
        const Node &node = *layout.nodes[i];

        PtNode snode = node.convert(box_transform);
        if (layout.first_child_ids[i] != u64_max) {
            for (u32 c = 0; c < 8; c++) {
                snode.childs_id[c] = layout.first_child_ids[i] + c;
            }
        }

        serial_tree[i]      = snode;
        linked_patch_ids[i] = node.linked_patchid;
        max_level           = sycl::max(max_level, node.tree_node.level);
    }
    level_count = max_level;

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform);
template void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform);

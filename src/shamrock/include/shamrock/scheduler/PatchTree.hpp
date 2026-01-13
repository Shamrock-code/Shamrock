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
 * @file PatchTree.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/patch/Patch.hpp"
#include "shamrock/scheduler/PatchTreeNode.hpp"
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace shamrock::scheduler {

    /**
     * @brief Patch Tree : Tree structure organisation for an abstract list of patches
     * Nb : this tree is compatible with multiple roots cf value in roots_id
     */
    class PatchTree {
        public:
        using Patch = patch::Patch;
        using Node  = scheduler::PatchTreeNode;

        /**
         * @brief set of root nodes ids
         */
        std::unordered_set<u64> roots_id;

        /**
         * @brief store the tree using a map
         *
         */
        std::unordered_map<u64, Node> tree;

        /**
         * @brief key of leaf nodes in tree
         *
         */
        std::unordered_set<u64> leaf_key;

        /**
         * @brief key of nodes that have only leafs as child
         *
         */
        std::unordered_set<u64> parent_of_only_leaf_key;

        /**
         * @brief split a leaf node
         *
         * @param id
         */
        void split_node(u64 id);

        /**
         * @brief merge childs of idparent (id parent should have only leafs as childs)
         *
         * @param idparent
         */
        void merge_node_dm1(u64 idparent);

        /**
         * @brief make tree from a patch table
         *
         * @param plist
         * @param max_val_1axis
         */
        [[deprecated]] void build_from_patchtable(std::vector<Patch> &plist, u64 max_val_1axis);

        /**
         * @brief update value in nodes (tree reduction)
         *
         * @param plist
         * @param id_patch_to_global_idx
         */
        void update_values_node(
            std::vector<Patch> &plist, const std::unordered_map<u64, u64> &id_patch_to_global_idx);

        /**
         * @brief update values in leafs and parent_of_only_leaf_key only
         *
         * @param plist
         * @param id_patch_to_global_idx
         */
        void partial_values_reduction(
            std::vector<Patch> &plist, const std::unordered_map<u64, u64> &id_patch_to_global_idx);

        /**
         * @brief Get list of nodes id to split
         *
         * @param crit_load_split
         * @return std::unordered_set<u64>
         */
        std::unordered_set<u64> get_split_request(u64 crit_load_split);

        /**
         * @brief Get list of nodes id to merge
         *
         * @param crit_load_merge
         * @return std::unordered_set<u64>
         */
        std::unordered_set<u64> get_merge_request(u64 crit_load_merge);

        void insert_root_node(u32 patch_id, patch::PatchCoord<3> coords);

        nlohmann::json serialize_patch_metadata() const;

        void load_json(const nlohmann::json &j);

        private:
        u64 next_id = 0;

        u64 insert_node(Node n);
        void remove_node(u64 id);

        void update_ptnode(
            Node &n,
            std::vector<Patch> &plist,
            const std::unordered_map<u64, u64> &id_patch_to_global_idx);
    };

    void to_json(nlohmann::json &j, const PatchTree &p);

    void from_json(const nlohmann::json &j, PatchTree &p);

} // namespace shamrock::scheduler

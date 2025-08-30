// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchTreeNode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/patch/PatchCoord.hpp"
#include "shamrock/patch/PatchCoordTransform.hpp"
#include <nlohmann/json.hpp>

namespace shamrock::scheduler {

    template<class vec>
    struct SerialPatchNode {
        vec box_min;
        vec box_max;
        std::array<u64, 8> childs_id;
    };

    /**
     * @brief Node information in the PatchTree link list
     */
    class LinkedTreeNode {
        public:
        u32 level;      ///< Level of the tree node
        u64 parent_nid; ///< Parent node id

        /**
         * @brief Array of childs node ids
         *
         * Childs nodes are stored in a fixed size array.
         *
         * If the node has no child the `[0] == u64_max`.
         * Otherwise the 8 childs ids are valid.
         */
        std::array<u64, 8> childs_nid{u64_max};

        bool is_leaf = true; ///< Is this node a leaf

        /**
         * @brief Store true if all childrens of this node are leafs
         *
         * This condition is mandatory for a merge to be possible
         */
        bool child_are_all_leafs = false;
    };

    /// Equal operator for LinkedTreeNode
    inline bool operator==(const LinkedTreeNode &lhs, const LinkedTreeNode &rhs) {
        return (lhs.level == rhs.level) && (lhs.parent_nid == rhs.parent_nid)
               && (lhs.childs_nid == rhs.childs_nid) && (lhs.is_leaf == rhs.is_leaf)
               && (lhs.child_are_all_leafs == rhs.child_are_all_leafs);
    }

    /**
     * @brief Node information in the patchtree + held patch info
     *
     */
    class PatchTreeNode {
        public:
        static constexpr u32 split_count = patch::PatchCoord<3>::splts_count;
        using PatchCoord                 = patch::PatchCoord<3>;

        PatchCoord patch_coord;

        LinkedTreeNode tree_node;
        u64 linked_patchid;

        // patch fields
        u64 load_value = u64_max;

        std::array<PatchTreeNode, split_count> get_split_nodes(u32 cur_id);

        bool is_leaf() { return tree_node.is_leaf; }

        u64 get_child_nid(u32 id) { return tree_node.childs_nid[id]; }

        /**
         * @brief Convert PatchTreeNode to SerialPatchNode using given coordinate transform
         *
         * @tparam vec The vector type
         * @param box_transform Coordinate transform from patch coordinate space to real one
         * @return SerialPatchNode<vec> Converted patch node
         */
        template<class vec>
        inline SerialPatchNode<vec> convert(
            const shamrock::patch::PatchCoordTransform<vec> box_transform) {
            SerialPatchNode<vec> n;

            // Convert patch range to object coordinates using given coordinate transform
            auto [bmin, bmax] = box_transform.to_obj_coord(patch_coord.get_patch_range());

            n.box_min      = bmin;
            n.box_max      = bmax;
            n.childs_id[0] = tree_node.childs_nid[0];
            n.childs_id[1] = tree_node.childs_nid[1];
            n.childs_id[2] = tree_node.childs_nid[2];
            n.childs_id[3] = tree_node.childs_nid[3];
            n.childs_id[4] = tree_node.childs_nid[4];
            n.childs_id[5] = tree_node.childs_nid[5];
            n.childs_id[6] = tree_node.childs_nid[6];
            n.childs_id[7] = tree_node.childs_nid[7];
            return n;
        }
    };

    /// Equal operator for PatchTreeNode
    inline bool operator==(const PatchTreeNode &lhs, const PatchTreeNode &rhs) {
        return (lhs.patch_coord == rhs.patch_coord) && (lhs.tree_node == rhs.tree_node)
               && (lhs.linked_patchid == rhs.linked_patchid) && (lhs.load_value == rhs.load_value);
    }

    inline auto PatchTreeNode::get_split_nodes(u32 cur_id)
        -> std::array<PatchTreeNode, split_count> {
        std::array<PatchCoord, split_count> splt_coord = patch_coord.split();

        std::array<PatchTreeNode, split_count> ret;

#pragma unroll
        for (u32 i = 0; i < split_count; i++) {
            ret[i].patch_coord          = splt_coord[i];
            ret[i].tree_node.level      = tree_node.level + 1;
            ret[i].tree_node.parent_nid = cur_id;
        }

        return ret;
    }

    /**
     * @brief Serializes a LinkedTreeNode object to a JSON object.
     *
     * @param j The JSON object to serialize to.
     * @param p The LinkedTreeNode object to serialize.
     */
    inline void to_json(nlohmann::json &j, const LinkedTreeNode &p) {

        j = nlohmann::json{
            {"level", p.level},
            {"parent_nid", p.parent_nid},
            {"childs_nid", p.childs_nid},
            {"is_leaf", p.is_leaf},
            {"child_are_all_leafs", p.child_are_all_leafs}};
    }

    /**
     * @brief Deserializes a JSON object to a LinkedTreeNode object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The LinkedTreeNode object to deserialize to.
     */
    inline void from_json(const nlohmann::json &j, LinkedTreeNode &p) {
        j.at("level").get_to(p.level);
        j.at("parent_nid").get_to(p.parent_nid);
        j.at("childs_nid").get_to(p.childs_nid);
        j.at("is_leaf").get_to(p.is_leaf);
        j.at("child_are_all_leafs").get_to(p.child_are_all_leafs);
    }

    /**
     * @brief Serializes a PatchTreeNode object to a JSON object.
     *
     * @param j The JSON object to serialize to.
     * @param p The PatchTreeNode object to serialize.
     */
    inline void to_json(nlohmann::json &j, const PatchTreeNode &p) {

        j = nlohmann::json{
            {"linked_patchid", p.linked_patchid},
            {"load_value", p.load_value},
            {"tree_node", p.tree_node},
            {"patch_coord",
             {
                 {"min", p.patch_coord.coord_min},
                 {"max", p.patch_coord.coord_max},
             }},
        };
    }

    /**
     * @brief Deserializes a JSON object to a PatchTreeNode object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The PatchTreeNode object to deserialize to.
     */
    inline void from_json(const nlohmann::json &j, PatchTreeNode &p) {
        j.at("linked_patchid").get_to(p.linked_patchid);
        j.at("load_value").get_to(p.load_value);
        j.at("tree_node").get_to(p.tree_node);
        j.at("patch_coord").at("min").get_to(p.patch_coord.coord_min);
        j.at("patch_coord").at("max").get_to(p.patch_coord.coord_max);
    }

} // namespace shamrock::scheduler

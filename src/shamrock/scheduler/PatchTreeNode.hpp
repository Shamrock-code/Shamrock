// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchTreeNode.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamrock/patch/PatchCoord.hpp"
#include "shamrock/patch/PatchCoordTransform.hpp"

namespace shamrock::scheduler {

    template<class vec>
    struct SerialPatchNode{
        vec box_min;
        vec box_max;
        std::array<u64,8> childs_id;
    };

    /**
     * @brief Node information in the PatchTree link list
     *
     */
    class LinkedTreeNode {
        public:
        u32 level;
        u64 parent_nid;
        std::array<u64,8> childs_nid {u64_max};

        bool is_leaf             = true;
        bool child_are_all_leafs = false;
    };

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

        std::array<PatchTreeNode, split_count> get_split_nodes();

        bool is_leaf(){
            return tree_node.is_leaf;
        }

        u64 get_child_nid(u32 id){
            return tree_node.childs_nid[id];
        }

        template<class vec>
        inline SerialPatchNode<vec> convert(const shamrock::patch::PatchCoordTransform<vec> box_transform){
            SerialPatchNode<vec> n;

            auto [bmin,bmax] = box_transform.to_obj_coord(patch_coord.get_patch_range());

            n.box_min    = bmin;
            n.box_max    = bmax;
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

    inline auto PatchTreeNode::get_split_nodes() -> std::array<PatchTreeNode, split_count> {
        std::array<PatchCoord, split_count> splt_coord = patch_coord.split();

        std::array<PatchTreeNode, split_count> ret;

        #pragma unroll
        for (u32 i = 0; i < split_count; i++) {
            ret[i].patch_coord = splt_coord[i];
            ret[i].tree_node.level = tree_node.level+1;
            ret[i].tree_node.parent_nid = tree_node.parent_nid;
        }

        return ret;
        
    }


    

} // namespace shamrock::scheduler
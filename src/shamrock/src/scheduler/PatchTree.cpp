// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchTree.cpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief PatchTree implementation
 *
 *
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace shamrock::scheduler {
    u64 PatchTree::insert_node(Node n) {
        tree[next_id] = n;
        next_id++;
        return next_id - 1;
    }

    void PatchTree::remove_node(u64 id) { tree.erase(id); }

    void PatchTree::split_node(u64 id) {

        leaf_key.erase(id);

        Node &curr = tree[id];

        auto &tree_node = curr.tree_node;
        if (tree_node.parent_nid != u64_max) {
            parent_of_only_leaf_key.erase(tree_node.parent_nid);
            tree[tree_node.parent_nid].tree_node.child_are_all_leafs = false;
        }

        tree_node.is_leaf             = false;
        tree_node.child_are_all_leafs = true;

        std::array<Node, Node::split_count> splitted_node = curr.get_split_nodes(id);

#pragma unroll
        for (u32 i = 0; i < Node::split_count; i++) {
            curr.tree_node.childs_nid[i] = insert_node(splitted_node[i]);
        }

        parent_of_only_leaf_key.insert(id);

#pragma unroll
        for (u32 i = 0; i < Node::split_count; i++) {
            leaf_key.insert(curr.tree_node.childs_nid[i]);
        }
    }

    void PatchTree::merge_node_dm1(u64 idparent) {

        Node &parent_node  = tree[idparent];
        auto &parent_tnode = parent_node.tree_node;

        if (!parent_tnode.child_are_all_leafs) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "node should be parent of only leafs");
        }

        leaf_key.erase(parent_tnode.childs_nid[0]);
        leaf_key.erase(parent_tnode.childs_nid[1]);
        leaf_key.erase(parent_tnode.childs_nid[2]);
        leaf_key.erase(parent_tnode.childs_nid[3]);
        leaf_key.erase(parent_tnode.childs_nid[4]);
        leaf_key.erase(parent_tnode.childs_nid[5]);
        leaf_key.erase(parent_tnode.childs_nid[6]);
        leaf_key.erase(parent_tnode.childs_nid[7]);

        remove_node(parent_tnode.childs_nid[0]);
        remove_node(parent_tnode.childs_nid[1]);
        remove_node(parent_tnode.childs_nid[2]);
        remove_node(parent_tnode.childs_nid[3]);
        remove_node(parent_tnode.childs_nid[4]);
        remove_node(parent_tnode.childs_nid[5]);
        remove_node(parent_tnode.childs_nid[6]);
        remove_node(parent_tnode.childs_nid[7]);

        parent_tnode.childs_nid[0] = u64_max;
        parent_tnode.childs_nid[1] = u64_max;
        parent_tnode.childs_nid[2] = u64_max;
        parent_tnode.childs_nid[3] = u64_max;
        parent_tnode.childs_nid[4] = u64_max;
        parent_tnode.childs_nid[5] = u64_max;
        parent_tnode.childs_nid[6] = u64_max;
        parent_tnode.childs_nid[7] = u64_max;

        leaf_key.insert(idparent);
        parent_tnode.is_leaf = true;

        parent_of_only_leaf_key.erase(idparent);
        parent_tnode.child_are_all_leafs = false;

        // check if parent of parent_tnode is parent of only leafs
        if (parent_tnode.parent_nid != u64_max) {
            bool only_leafs = true;

            Node &parent_node = tree[parent_tnode.parent_nid];

            for (u8 idc = 0; idc < 8; idc++) {
                only_leafs = only_leafs && tree[parent_node.get_child_nid(idc)].is_leaf();
            }

            if (only_leafs) {
                parent_node.tree_node.child_are_all_leafs = true;
                parent_of_only_leaf_key.insert(parent_tnode.parent_nid);
            }
        }
    }

    void PatchTree::insert_root_node(u32 patch_id, patch::PatchCoord<3> coords) {
        Node root;
        root.patch_coord                   = coords;
        root.tree_node.level               = 0;
        root.tree_node.parent_nid          = u64_max;
        root.tree_node.is_leaf             = true;
        root.tree_node.child_are_all_leafs = false;
        root.linked_patchid                = patch_id;

        u64 root_id = insert_node(root);
        leaf_key.insert(root_id);
        roots_id.insert(root_id);
    }

    void PatchTree::build_from_patchtable(
        std::vector<shamrock::patch::Patch> &plist, u64 max_val_1axis) {

        if (plist.size() > 1) {
            Node root;
            root.patch_coord.coord_max[0] = max_val_1axis;
            root.patch_coord.coord_max[1] = max_val_1axis;
            root.patch_coord.coord_max[2] = max_val_1axis;
            root.patch_coord.coord_min[0] = 0;
            root.patch_coord.coord_min[1] = 0;
            root.patch_coord.coord_min[2] = 0;
            root.tree_node.level          = 0;
            root.tree_node.parent_nid     = u64_max;

            u64 root_id = insert_node(root);
            leaf_key.insert(root_id);
            roots_id.insert(root_id);

            std::vector<u64> complete_vec;
            for (u64 i = 0; i < plist.size(); i++) {
                complete_vec.push_back(i);
            }

            std::vector<std::tuple<u64, std::vector<u64>>> tree_vec(1);

            tree_vec[0] = {root_id, complete_vec};

            while (tree_vec.size() > 0) {
                std::vector<std::tuple<u64, std::vector<u64>>> next_tree_vec;
                for (auto &[idtree, idvec] : tree_vec) {

                    Node &ptn = tree[idtree];

                    split_node(idtree);

                    for (u8 child_id = 0; child_id < 8; child_id++) {

                        u64 ptnode_id = ptn.tree_node.childs_nid[child_id];
                        std::vector<u64> buf;

                        auto &curr = tree[ptnode_id].patch_coord;

                        for (u64 idxptch : idvec) {
                            shamrock::patch::Patch &p = plist[idxptch];

                            bool is_inside = BBAA::iscellb_inside_a<u64_3>(
                                {curr.coord_min[0], curr.coord_min[1], curr.coord_min[2]},
                                {curr.coord_max[0], curr.coord_max[1], curr.coord_max[2]},
                                {p.coord_min[0], p.coord_min[1], p.coord_min[2]},
                                {p.coord_max[0], p.coord_max[1], p.coord_max[2]});

                            if (is_inside) {
                                buf.push_back(idxptch);
                            }

                            /*
                            std::cout << " ( " <<
                                "[" << curr.x_min << "," << curr.x_max << "] " <<
                                "[" << curr.y_min << "," << curr.y_max << "] " <<
                                "[" << curr.z_min << "," << curr.z_max << "] " <<
                                " )  node : ( " <<
                                "[" << p.x_min << "," << p.x_max << "] " <<
                                "[" << p.y_min << "," << p.y_max << "] " <<
                                "[" << p.z_min << "," << p.z_max << "] " <<
                                " ) "<< is_inside;

                            if(is_inside){ std::cout << " -> push " << idxptch;}

                            std::cout << std::endl;
                            */
                        }

                        if (buf.size() == 1) {
                            // std::cout << "set linked id node " << buf[0] << " : "  <<
                            // plist[buf[0]].id_patch << std::endl;
                            tree[ptnode_id].linked_patchid = plist[buf[0]].id_patch;
                        } else {
                            next_tree_vec.push_back({ptnode_id, buf});
                        }
                    }

                } // std::cout << "----------------" << std::endl;

                tree_vec = next_tree_vec;
            }
        } else if (plist.size() == 1) {

            patch::PatchCoord<3> patch_coord;
            patch_coord.coord_max[0] = max_val_1axis;
            patch_coord.coord_max[1] = max_val_1axis;
            patch_coord.coord_max[2] = max_val_1axis;
            patch_coord.coord_min[0] = 0;
            patch_coord.coord_min[1] = 0;
            patch_coord.coord_min[2] = 0;
            insert_root_node(plist[0].id_patch, patch_coord);
        }
    }

    void PatchTree::update_ptnode(
        Node &n,
        std::vector<shamrock::patch::Patch> &plist,
        const std::unordered_map<u64, u64> &id_patch_to_global_idx) {

        auto &tnode = n.tree_node;

        if (n.linked_patchid != u64_max) {
            n.load_value = 0;
            n.load_value += plist[id_patch_to_global_idx.at(n.linked_patchid)].load_value;
        } else if (tnode.childs_nid[0] != u64_max) {

            bool has_err_val = false;

            n.load_value = 0;
            for (u8 idc = 0; idc < 8; idc++) {

                if (tree[tnode.childs_nid[idc]].load_value == u64_max)
                    has_err_val = true;

                n.load_value += tree[tnode.childs_nid[idc]].load_value;
            }

            if (has_err_val) {
                n.load_value = u64_max;
            }
        }
    }

    // TODO add test value on root = sum all leaf
    void PatchTree::update_values_node(
        std::vector<shamrock::patch::Patch> &plist,
        const std::unordered_map<u64, u64> &id_patch_to_global_idx) {

        tree[0].load_value = u64_max;

        while (tree[0].load_value == u64_max) {
            for (auto &[key, ptnode] : tree) {
                update_ptnode(ptnode, plist, id_patch_to_global_idx);
            }
        }
    }

    void PatchTree::partial_values_reduction(
        std::vector<shamrock::patch::Patch> &plist,
        const std::unordered_map<u64, u64> &id_patch_to_global_idx) {
        StackEntry stack_loc{};

        for (u64 id_leaf : leaf_key) {
            update_ptnode(tree[id_leaf], plist, id_patch_to_global_idx);
        }

        for (u64 id_leaf : parent_of_only_leaf_key) {
            update_ptnode(tree[id_leaf], plist, id_patch_to_global_idx);
        }
    }

    std::unordered_set<u64> PatchTree::get_split_request(u64 crit_load_split) {
        StackEntry stack_loc{};
        std::unordered_set<u64> rq;

        for (u64 a : leaf_key) {
            if (tree[a].load_value > crit_load_split) {
                rq.insert(a);
            }
        }

        return rq;
    }

    std::unordered_set<u64> PatchTree::get_merge_request(u64 crit_load_merge) {
        StackEntry stack_loc{};
        std::unordered_set<u64> rq;

        for (u64 a : parent_of_only_leaf_key) {
            if (tree[a].load_value < crit_load_merge) {
                rq.insert(a);
            }
        }

        return rq;
    }

    /**
     * @brief Serialize the metadata of the patch tree.
     *
     * @return The serialized metadata as a JSON object.
     */
    nlohmann::json PatchTree::serialize_patch_metadata() const {
        // Serialized fields :
        // - std::unordered_set<u64> roots_id;
        // - std::unordered_map<u64, Node> tree;
        // - std::unordered_set<u64> leaf_key;
        // - std::unordered_set<u64> parent_of_only_leaf_key;

        // Note it is not possible to serialize STL containers if this function is not marked const
        return {
            {"roots_id", roots_id},
            {"tree", tree},
            {"leaf_key", leaf_key},
            {"parent_of_only_leaf_key", parent_of_only_leaf_key},
            {"next_id", next_id}};
    }

    /**
     * @brief Load the metadata of the patch tree from a JSON object.
     *
     * @param j The JSON object containing the metadata.
     */
    void PatchTree::load_json(const nlohmann::json &j) {

        j.at("roots_id").get_to(roots_id);
        j.at("tree").get_to(tree);
        j.at("leaf_key").get_to(leaf_key);
        j.at("parent_of_only_leaf_key").get_to(parent_of_only_leaf_key);
        j.at("next_id").get_to(next_id);
    }

    /**
     * @brief Serializes a PatchTree object into a JSON object.
     *
     * @param j The JSON object to serialize the PatchTree into.
     * @param p The PatchTree object to serialize.
     */
    void to_json(nlohmann::json &j, const PatchTree &p) { j = p.serialize_patch_metadata(); }

    /**
     * @brief Deserializes a JSON object into a PatchTree object.
     *
     * @param j The JSON object to deserialize.
     * @param p The PatchTree object to deserialize into.
     */
    void from_json(const nlohmann::json &j, PatchTree &p) { p.load_json(j); }

} // namespace shamrock::scheduler

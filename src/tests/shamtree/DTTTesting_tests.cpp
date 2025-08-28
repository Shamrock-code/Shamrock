// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <vector>

using Tmorton = u64;
using Tvec    = f64_3;
using Tscal   = shambase::VecComponent<Tvec>;

using ObjItHost = shamtree::CLBVHObjectIteratorHost<Tmorton, Tvec, 3>;

inline bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
    Tvec s_a      = (a.upper - a.lower);
    Tvec s_b      = (b.upper - b.lower);
    Tvec r_a      = (a.upper + a.lower) / 2;
    Tvec r_b      = (b.upper + b.lower) / 2;
    Tvec delta_ab = r_a - r_b;

    Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

    if (delta_ab_sq == 0) {
        return false;
    }

    Tscal s_a_sq = sham::dot(s_a, s_a);
    Tscal s_b_sq = sham::dot(s_b, s_b);

    Tscal r_a_sq = sham::dot(r_a, r_a);
    Tscal r_b_sq = sham::dot(r_b, r_b);

    Tscal theta_sq = (s_a_sq + s_b_sq) / delta_ab_sq;

    return theta_sq < theta_crit * theta_crit;
}

inline void dtt_recursive_internal(
    u32 cell_a,
    u32 cell_b,
    const ObjItHost::acc &acc,
    Tscal theta_crit,
    std::vector<std::pair<u32, u32>> &internal_node_interactions,
    std::vector<std::pair<u32, u32>> &unrolled_interact) {

    auto dtt_child_call = [&](u32 cell_a, u32 cell_b) {
        dtt_recursive_internal(
            cell_a, cell_b, acc, theta_crit, internal_node_interactions, unrolled_interact);
    };

    auto &ttrav = acc.tree_traverser.tree_traverser;

    Tvec aabb_min_a = acc.tree_traverser.aabb_min[cell_a];
    Tvec aabb_max_a = acc.tree_traverser.aabb_max[cell_a];

    Tvec aabb_min_b = acc.tree_traverser.aabb_min[cell_b];
    Tvec aabb_max_b = acc.tree_traverser.aabb_max[cell_b];

    shammath::AABB<Tvec> aabb_a = {aabb_min_a, aabb_max_a};
    shammath::AABB<Tvec> aabb_b = {aabb_min_b, aabb_max_b};

    bool crit = mac(aabb_a, aabb_b, theta_crit) == false;

    if (crit) {

        u32 child_a_1 = ttrav.get_left_child(cell_a);
        u32 child_a_2 = ttrav.get_right_child(cell_a);
        u32 child_b_1 = ttrav.get_left_child(cell_b);
        u32 child_b_2 = ttrav.get_right_child(cell_b);

        bool child_a_1_leaf = ttrav.is_id_leaf(child_a_1);
        bool child_a_2_leaf = ttrav.is_id_leaf(child_a_2);
        bool child_b_1_leaf = ttrav.is_id_leaf(child_b_1);
        bool child_b_2_leaf = ttrav.is_id_leaf(child_b_2);

        if (child_a_1_leaf || child_a_2_leaf || child_b_1_leaf || child_b_2_leaf) {
            unrolled_interact.push_back({cell_a, cell_b});
            return;
        }

        dtt_child_call(child_a_1, child_b_1);
        dtt_child_call(child_a_2, child_b_1);
        dtt_child_call(child_a_1, child_b_2);
        dtt_child_call(child_a_2, child_b_2);

    } else {
        internal_node_interactions.push_back({cell_a, cell_b});
    }
}

inline void dtt_recursive_ref(
    const sham::DeviceBuffer<Tvec> &positions,
    const shamtree::CompressedLeafBVH<Tmorton, Tvec, 3> &bvh,
    Tscal theta_crit,
    std::vector<std::pair<u32, u32>> &internal_node_interactions,
    std::vector<std::pair<u32, u32>> &unrolled_interact) {

    auto obj_it_host = bvh.get_object_iterator_host();
    auto acc         = obj_it_host.get_read_access();

    dtt_recursive_internal(0, 0, acc, theta_crit, internal_node_interactions, unrolled_interact);
}

inline void validate_dtt_results(
    const sham::DeviceBuffer<Tvec> &positions,
    const shamtree::CompressedLeafBVH<Tmorton, Tvec, 3> &bvh,
    Tscal theta_crit,
    std::vector<std::pair<u32, u32>> &internal_node_interactions,
    std::vector<std::pair<u32, u32>> &unrolled_interact) {

    u32 Npart    = positions.get_size();
    u32 Npart_sq = Npart * Npart;

    logger::raw_ln(
        "node/node               :",
        internal_node_interactions.size(),
        " ratio :",
        (double) internal_node_interactions.size() / Npart_sq);
    logger::raw_ln(
        "P2P                     :",
        unrolled_interact.size(),
        " ratio :",
        (double) unrolled_interact.size() / Npart_sq);

    shamtree::CellIteratorHost cell_it_bind = bvh.get_cell_iterator_host();
    auto cell_it                            = cell_it_bind.get_read_access();

    std::vector<std::tuple<u32, u32>> part_interact_node_node{};
    std::vector<std::tuple<u32, u32>> part_interact_leaf_leaf{};

    for (auto [cell_a, cell_b] : internal_node_interactions) {

        u32 node_a = cell_a;
        u32 node_b = cell_b;

        cell_it.for_each_in_cell(node_a, [&](u32 id_a) {
            cell_it.for_each_in_cell(node_b, [&](u32 id_b) {
                part_interact_node_node.push_back({id_a, id_b});
            });
        });
    }

    for (auto [cell_a, cell_b] : unrolled_interact) {

        u32 leaf_a = cell_a;
        u32 leaf_b = cell_b;

        cell_it.for_each_in_cell(leaf_a, [&](u32 id_a) {
            cell_it.for_each_in_cell(leaf_b, [&](u32 id_b) {
                part_interact_leaf_leaf.push_back({id_a, id_b});
            });
        });
    }

    logger::raw_ln(
        "part interact node/node :",
        part_interact_node_node.size(),
        " ratio :",
        (double) part_interact_node_node.size() / Npart_sq);
    logger::raw_ln(
        "part interact leaf/leaf :",
        part_interact_leaf_leaf.size(),
        " ratio :",
        (double) part_interact_leaf_leaf.size() / Npart_sq);

    logger::raw_ln("sum :", part_interact_node_node.size() + part_interact_leaf_leaf.size());

    std::set<std::pair<u32, u32>> part_interact{};
    // insert both sets
    for (auto [id_a, id_b] : part_interact_node_node) {
        part_interact.insert({id_a, id_b});
    }
    for (auto [id_a, id_b] : part_interact_leaf_leaf) {
        part_interact.insert({id_a, id_b});
    }

    u32 missing_pairs = 0;
    // now check that all pairs exist in that list
    for (u32 i = 0; i < Npart; i++) {
        for (u32 j = 0; j < Npart; j++) {
            if (part_interact.find({i, j}) == part_interact.end()) {
                logger::raw_ln("pair not found :", i, j);
                missing_pairs++;
            }
        }
    }

    REQUIRE_EQUAL(missing_pairs, 0);
    REQUIRE_EQUAL(part_interact.size(), Npart_sq);
}

TestStart(Unittest, "DTT_testing1", dtt_testing1, 1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    u32 Npart           = 1000;
    u32 Npart_sq        = Npart * Npart;
    u32 reduction_level = 0;
    Tscal theta_crit    = 0.5;

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>({-1, -1, -1}, {1, 1, 1});

    // build a vector of random positions
    std::vector<Tvec> positions
        = shamalgs::primitives::mock_vector<Tvec>(0x111, Npart, bb.lower, bb.upper);

    sham::DeviceBuffer<Tvec> partpos_buf(positions.size(), dev_sched);
    partpos_buf.copy_from_stdvec(positions);

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    bvh.rebuild_from_positions(partpos_buf, bb, reduction_level);

    auto obj_it_host = bvh.get_object_iterator_host();

    {
        std::vector<std::pair<u32, u32>> internal_node_interactions{};
        std::vector<std::pair<u32, u32>> unrolled_interact{};
        dtt_recursive_ref(
            partpos_buf, bvh, theta_crit, internal_node_interactions, unrolled_interact);

        validate_dtt_results(
            partpos_buf, bvh, theta_crit, internal_node_interactions, unrolled_interact);
    }
}

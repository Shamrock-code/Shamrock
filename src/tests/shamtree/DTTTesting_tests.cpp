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
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <vector>
#include "shambackends/fmt_bindings/fmt_defs.hpp"

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

TestStart(Unittest, "DTT_testing1", dtt_testing1, 1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    u32 Npart           = 1000;
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
        std::vector<std::pair<u32, u32>> unrolled_interac{};
        dtt_recursive_ref(
            partpos_buf, bvh, theta_crit, internal_node_interactions, unrolled_interac);
        logger::raw_ln("node/node :",internal_node_interactions.size());
        logger::raw_ln("P2P       :",unrolled_interac.size());
    }

}

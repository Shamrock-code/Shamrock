// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/StlContainerConversion.hpp"
#include "shambase/time.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/details/dtt_parallel_select.hpp"
#include "shamtree/details/dtt_reference.hpp"
#include "shamtree/details/dtt_scan_multipass.hpp"
#include <set>
#include <vector>

using Tmorton = u64;
using Tvec    = f64_3;
using Tscal   = shambase::VecComponent<Tvec>;

inline void validate_dtt_results(
    const sham::DeviceBuffer<Tvec> &positions,
    const shamtree::CompressedLeafBVH<Tmorton, Tvec, 3> &bvh,
    Tscal theta_crit,
    shamtree::DTTResult &result,
    bool ordered_result) {

    __shamrock_stack_entry();

    std::vector<u32_2> internal_node_interactions = result.node_interactions_m2m.copy_to_stdvec();
    std::vector<u32_2> unrolled_interact          = result.node_interactions_p2p.copy_to_stdvec();

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

    if (ordered_result) {
        logger::raw_ln("testing ordered result");

        REQUIRE(result.is_ordered());
        REQUIRE(bool(result.ordered_result));

        if (result.ordered_result.has_value()) {

            auto test_ordered = [](std::vector<u32_2> &part_interact) {
                u32 offenses = 0;

                u32 last_id_a = 0;
                for (auto r : part_interact) {
                    u32 id_a = r.x();
                    u32 id_b = r.y();

                    if (id_a < last_id_a) {
                        offenses++;
                    }

                    last_id_a = id_a;
                }
                return offenses;
            };

            REQUIRE_EQUAL(test_ordered(unrolled_interact), 0);
            REQUIRE_EQUAL(test_ordered(internal_node_interactions), 0);

            auto m2m = result.node_interactions_m2m.copy_to_stdvec();
            auto p2p = result.node_interactions_p2p.copy_to_stdvec();

            auto m2m_offset = result.ordered_result->offset_m2m.copy_to_stdvec();
            auto p2p_offset = result.ordered_result->offset_p2p.copy_to_stdvec();

            // test that if I look in slot i I find only stuff with i in x part

            u32 m2m_sorting_offenses = 0;
            for (u32 i = 0; i < Npart; i++) {
                for (u32 j = m2m_offset[i]; j < m2m_offset[i + 1]; j++) {
                    u32_2 interact = m2m[j];
                    if (interact.x() != i) {
                        m2m_sorting_offenses++;
                    }
                }
            }

            u32 p2p_sorting_offenses = 0;
            for (u32 i = 0; i < Npart; i++) {
                for (u32 j = p2p_offset[i]; j < p2p_offset[i + 1]; j++) {
                    u32_2 interact = p2p[j];
                    if (interact.x() != i) {
                        p2p_sorting_offenses++;
                    }
                }
            }

            REQUIRE_EQUAL(m2m_sorting_offenses, 0);
            REQUIRE_EQUAL(p2p_sorting_offenses, 0);
        }

    } else {
        REQUIRE(!result.is_ordered());
        REQUIRE(!bool(result.ordered_result));
    }

    for (auto r : internal_node_interactions) {

        u32 node_a = r.x();
        u32 node_b = r.y();

        cell_it.for_each_in_cell(node_a, [&](u32 id_a) {
            cell_it.for_each_in_cell(node_b, [&](u32 id_b) {
                part_interact_node_node.push_back({id_a, id_b});
            });
        });
    }

    for (auto r : unrolled_interact) {

        u32 leaf_a = r.x();
        u32 leaf_b = r.y();

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
                // logger::raw_ln("pair not found :", i, j);
                missing_pairs++;
            }
        }
    }

    REQUIRE_EQUAL(missing_pairs, 0);
    REQUIRE_EQUAL(part_interact.size(), Npart_sq);
}

void dtt_test(u32 Npart, u32 reduction_level, Tscal theta_crit, bool ordered_result) {
    __shamrock_stack_entry();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>({-1, -1, -1}, {1, 1, 1});

    // build a vector of random positions
    std::vector<Tvec> positions
        = shamalgs::primitives::mock_vector<Tvec>(0x111, Npart, bb.lower, bb.upper);

    sham::DeviceBuffer<Tvec> partpos_buf(positions.size(), dev_sched);
    partpos_buf.copy_from_stdvec(positions);

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    bvh.rebuild_from_positions(partpos_buf, bb, reduction_level);

    std::vector<u32_2> m2m_ref{};
    std::vector<u32_2> p2p_ref{};

    auto equals_unordered = [](std::vector<u32_2> a, std::vector<u32_2> b) -> bool {
        if (a.size() != b.size()) {
            return false;
        }

        auto comp = [](const u32_2 &v1, const u32_2 &v2) {
            if (v1.x() != v2.x())
                return v1.x() < v2.x();
            return v1.y() < v2.y();
        };

        std::sort(a.begin(), a.end(), comp);
        std::sort(b.begin(), b.end(), comp);

        return sham::equals(a, b);
    };

    {
        shambase::Timer timer;
        timer.start();
        auto result = shamtree::details::DTTCpuReference<Tmorton, Tvec, 3>::dtt(
            shamsys::instance::get_compute_scheduler_ptr(), bvh, theta_crit, ordered_result);
        timer.end();
        logger::raw_ln("DTTCpuReference :", timer.get_time_str());

        validate_dtt_results(partpos_buf, bvh, theta_crit, result, ordered_result);

        m2m_ref = result.node_interactions_m2m.copy_to_stdvec();
        p2p_ref = result.node_interactions_p2p.copy_to_stdvec();
    }

    auto current_impl = shamtree::impl::get_current_impl_clbvh_dual_tree_traversal_impl();

    for (auto impl : shamtree::impl::get_default_impl_list_clbvh_dual_tree_traversal()) {
        shamtree::impl::set_impl_clbvh_dual_tree_traversal(impl.impl_name, impl.params);

        shambase::Timer timer;
        timer.start();
        auto result = shamtree::clbvh_dual_tree_traversal(
            shamsys::instance::get_compute_scheduler_ptr(), bvh, theta_crit, ordered_result);
        timer.end();
        logger::raw_ln(impl.impl_name, " :", timer.get_time_str());

        validate_dtt_results(partpos_buf, bvh, theta_crit, result, ordered_result);

        std::vector<u32_2> internal_node_interactions
            = result.node_interactions_m2m.copy_to_stdvec();
        std::vector<u32_2> unrolled_interact = result.node_interactions_p2p.copy_to_stdvec();

        REQUIRE_EQUAL_CUSTOM_COMP(internal_node_interactions, m2m_ref, equals_unordered);
        REQUIRE_EQUAL_CUSTOM_COMP(unrolled_interact, p2p_ref, equals_unordered);
    }

    // reset to current impl
    shamtree::impl::set_impl_clbvh_dual_tree_traversal(current_impl.impl_name, current_impl.params);
}

inline void dtt_test_empty(bool ordered_result) {
    __shamrock_stack_entry();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();
    auto bvh       = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    auto current_impl = shamtree::impl::get_current_impl_clbvh_dual_tree_traversal_impl();

    for (auto impl : shamtree::impl::get_default_impl_list_clbvh_dual_tree_traversal()) {
        shamtree::impl::set_impl_clbvh_dual_tree_traversal(impl.impl_name, impl.params);

        auto run_dtt = [&]() {
            shamtree::clbvh_dual_tree_traversal(
                shamsys::instance::get_compute_scheduler_ptr(), bvh, 0.5, ordered_result);
        };

        REQUIRE_EXCEPTION_THROW(run_dtt(), std::invalid_argument);
    }
    // reset to current impl
    shamtree::impl::set_impl_clbvh_dual_tree_traversal(current_impl.impl_name, current_impl.params);
}

inline void dtt_tests(bool ordered_result) {
    __shamrock_stack_entry();
    dtt_test(1000, 1, 0.5, ordered_result);
    dtt_test(1000, 1, 0.0, ordered_result);
    dtt_test(1, 1, 0.5, ordered_result);
    dtt_test_empty(ordered_result);
}

TestStart(Unittest, "shamtree::clbvh_dual_tree_traversal(unordered)", dtt_testing1, 1) {
    dtt_tests(false);
}

TestStart(Unittest, "shamtree::clbvh_dual_tree_traversal(ordered)", dtt_testing2, 1) {
    dtt_tests(true);
}

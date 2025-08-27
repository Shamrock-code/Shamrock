// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CLBVHObjectIterator.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <vector>

using Tmorton = u64;
using Tvec    = f64_3;
using Tscal   = shambase::VecComponent<Tvec>;

TestStart(Unittest, "shamtree/CellIterator", test_cell_iterator, 1) {

    std::vector<Tvec> partpos{
        Tvec(0, 0, 0),
        Tvec(0.1, 0.0, 0.0),
        Tvec(0.0, 0.1, 0.0),
        Tvec(0.0, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.0),
        Tvec(0.0, 0.1, 0.1),
        Tvec(0.1, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.1),
        Tvec(0.2, 0.2, 0.2),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4),
        Tvec(1, 1, 1),
        Tvec(2, 2, 2),
        Tvec(-1, -1, -1)};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(partpos.size(), dev_sched);

    partpos_buf.copy_from_stdvec(partpos);

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    bvh.rebuild_from_positions(partpos_buf, bb, 1);

    auto obj_it_host  = bvh.get_object_iterator_host();
    auto cell_it_host = bvh.get_cell_iterator_host();

    auto bvh_it  = obj_it_host.get_read_access();
    auto cell_it = cell_it_host.get_read_access();

    auto compare_cell_ids = [&](u32 cell_id) {
        // from cell iterator
        std::vector<u32> result{};

        bool is_leaf = bvh_it.tree_traverser.tree_traverser.is_id_leaf(cell_id);

        u32 internal_cell_count = bvh_it.tree_traverser.tree_traverser.offset_leaf;

        cell_it.for_each_in_cell(cell_id, internal_cell_count, is_leaf, [&](u32 obj_id) {
            result.push_back(obj_id);
        });

        // from traverser
        std::vector<u32> result2{};

        if (is_leaf) {
            u32 leaf_id = cell_id - bvh_it.tree_traverser.tree_traverser.offset_leaf;
            bvh_it.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 obj_id) {
                result2.push_back(obj_id);
            });
        } else {
            bvh_it.tree_traverser.traverse_tree_base(
                cell_id,
                [&](u32 node_id) {
                    return true;
                },
                [&](u32 node_id) {
                    u32 leaf_id = node_id - bvh_it.tree_traverser.tree_traverser.offset_leaf;
                    bvh_it.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 obj_id) {
                        result2.push_back(obj_id);
                    });
                },
                [&](u32) {});
        }

        i32 rchild = (is_leaf) ? -1 : bvh_it.tree_traverser.tree_traverser.get_right_child(cell_id);
        i32 lchild = (is_leaf) ? -1 : bvh_it.tree_traverser.tree_traverser.get_left_child(cell_id);

        // if you want to visualize of the ids are mapped, uncomment this
        // logger::raw_ln(cell_id, lchild, rchild, result);

        REQUIRE_EQUAL(result, result2);
    };

    u32 cell_count = bvh.get_total_cell_count();

    for (u32 cell_id = 0; cell_id < cell_count; cell_id++) {
        compare_cell_ids(cell_id);
    }
}

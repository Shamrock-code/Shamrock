// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamtree/MortonReducedSet.hpp"
#include <vector>

using Tval    = double;
using Tmorton = u64;
using Tvec    = f64_3;

TestStart(Unittest, "shamtree/KarrasRadixTreeField", test_karras_radix_tree_field_max_field, 1) {
    // Use the same 11-particle partpos as in KarrasRadixTreeAABBTests.cpp
    std::vector<Tval> field_values{1.0, 5.0, 3.0, 2.0, 8.0, 7.0, 4.0, 6.0, 9.0, 10.0, 11.0};
    std::vector<Tvec> partpos{
        Tvec(0.0, 0.0, 0.0),
        Tvec(0.1, 0.0, 0.0),
        Tvec(0.0, 0.1, 0.0),
        Tvec(0.0, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.0),
        Tvec(0.0, 0.1, 0.1),
        Tvec(0.1, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.1),
        Tvec(0.2, 0.2, 0.2),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4)};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceBuffer<Tval> field_buf(field_values.size(), dev_sched);
    field_buf.copy_from_stdvec(field_values);
    sham::DeviceBuffer<Tvec> partpos_buf(partpos.size(), dev_sched);
    partpos_buf.copy_from_stdvec(partpos);

    // Use CLBVH to get the tree and cell iterator
    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));
    auto bvh                = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);
    bvh.rebuild_from_positions(partpos_buf, bb, 1);
    auto &tree   = bvh.structure;
    auto cell_it = bvh.reduced_morton_set.get_leaf_cell_iterator();

    // Prepare output buffer
    auto tree_field = shamtree::new_empty_karras_radix_tree_field<Tval>();

    // Compute max field per cell
    auto result_field = shamtree::compute_tree_field_max_field<Tval>(
        tree, cell_it, std::move(tree_field), field_buf);

    // Download result
    std::vector<Tval> result = result_field.buf_field.copy_to_stdvec();

    // check tree structure

    // 1.0, 5.0, 3.0, 2.0, 8.0, 7.0, 4.0, 6.0, 9.0, 10.0, 11.0

    std::vector<u32> index_map_obj_idx = {0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 15, 11, 14, 12, 13};

    std::vector<Tmorton> test_mortons_sorted
        = {0_u64,                   // 1.0
           2533893416718345_u64,    // 2.0
           5067786833436690_u64,    // 3.0
           7601680250155035_u64,    // 7.0
           10135573666873380_u64,   // 5.0
           12669467083591725_u64,   // 4.0
           15203360500310070_u64,   // 8.0
           17737253917028415_u64,   // 6.0
           141898031336227320_u64,  // 9.0
           1011023473270619655_u64, // 10.0
           1135184250689818560_u64, // 11.0
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64};

    std::vector<Tmorton> reduced_morton_codes = {
        0_u64,                  // 1.0, 2.0
        5067786833436690_u64,   // 3.0, 7.0
        10135573666873380_u64,  // 5.0, 4.0
        15203360500310070_u64,  // 8.0, 6.0
        141898031336227320_u64, // 9.0
        1011023473270619655_u64 // 10.0, 11.0
    };

    std::vector<u32> buf_reduc_index_map = {0, 2, 4, 6, 8, 9, 11, 0};

    std::vector<u32> expected_lchild_id  = {4, 0, 2, 1, 3};
    std::vector<u8> expected_lchild_flag = {0, 1, 1, 0, 0};
    std::vector<u32> expected_rchild_id  = {5, 1, 3, 2, 4};
    std::vector<u8> expected_rchild_flag = {1, 1, 1, 0, 1};
    std::vector<u32> expected_endrange   = {5, 0, 3, 0, 0};

    /*
    l0 = 2.0
    l1 = 7.0
    l2 = 5.0
    l3 = 8.0
    l4 = 9.0
    l5 = 11.0

    digraph G {
    rankdir=LR;
    i0 -> i4;
    i0 -> l5;
    i1 -> l0;
    i1 -> l1;
    i2 -> l2;
    i2 -> l3;
    i3 -> i1;
    i3 -> i2;
    i4 -> i3;
    i4 -> l4;
    }

    i1 = 7.0
    i2 = 8.0
    i3 = 8.0
    i4 = 9.0
    i0 = 11.0
    */

    std::vector<Tval> expected_result = {
        11, // i0
        7,  // i1
        8,  // i2
        8,  // i3
        9,  // i4
        2,  // l0
        7,  // l1
        5,  // l2
        8,  // l3
        9,  // l4
        11  // l5
    };

    REQUIRE_EQUAL(bvh.reduced_morton_set.morton_codes_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(
        bvh.reduced_morton_set.morton_codes_set.sorted_morton_codes.copy_to_stdvec(),
        test_mortons_sorted);
    REQUIRE_EQUAL(bvh.reduced_morton_set.morton_codes_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(
        bvh.reduced_morton_set.morton_codes_set.map_morton_id_to_obj_id.copy_to_stdvec(),
        index_map_obj_idx);

    REQUIRE_EQUAL(bvh.reduced_morton_set.reduce_code_count, 6);
    REQUIRE_EQUAL(bvh.reduced_morton_set.reduced_morton_codes.get_size(), 6);
    REQUIRE_EQUAL(
        bvh.reduced_morton_set.reduced_morton_codes.copy_to_stdvec(), reduced_morton_codes);
    REQUIRE_EQUAL(bvh.reduced_morton_set.buf_reduc_index_map.get_size(), 6 + 2);
    REQUIRE_EQUAL(bvh.reduced_morton_set.buf_reduc_index_map.copy_to_stdvec(), buf_reduc_index_map);

    REQUIRE_EQUAL(tree.buf_lchild_id.copy_to_stdvec(), expected_lchild_id);
    REQUIRE_EQUAL(tree.buf_rchild_id.copy_to_stdvec(), expected_rchild_id);
    REQUIRE_EQUAL(tree.buf_lchild_flag.copy_to_stdvec(), expected_lchild_flag);
    REQUIRE_EQUAL(tree.buf_rchild_flag.copy_to_stdvec(), expected_rchild_flag);
    REQUIRE_EQUAL(tree.buf_endrange.copy_to_stdvec(), expected_endrange);

    REQUIRE_EQUAL(tree.buf_lchild_id.get_size(), 5);
    REQUIRE_EQUAL(tree.buf_rchild_id.get_size(), 5);
    REQUIRE_EQUAL(tree.buf_lchild_flag.get_size(), 5);
    REQUIRE_EQUAL(tree.buf_rchild_flag.get_size(), 5);
    REQUIRE_EQUAL(tree.buf_endrange.get_size(), 5);

    REQUIRE_EQUAL(result.size(), 11);
    REQUIRE_EQUAL(result, expected_result);
}

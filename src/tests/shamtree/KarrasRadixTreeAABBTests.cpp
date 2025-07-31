// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/KarrasRadixTreeAABB.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamtree/MortonReducedSet.hpp"
#include <vector>

using Tvec    = f64_3;
using Tmorton = u64;

TestStart(Unittest, "shamtree/KarrasRadixTreeAABB", test_karras_radix_tree_aabb, 1) {

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

    std::vector<Tmorton> test_mortons
        = {0_u64,                   // (0.0, 0.0, 0.0)
           10135573666873380_u64,   // (0.1, 0.0, 0.0)
           5067786833436690_u64,    // (0.0, 0.1, 0.0)
           2533893416718345_u64,    // (0.0, 0.0, 0.1)
           15203360500310070_u64,   // (0.1, 0.1, 0.0)
           7601680250155035_u64,    // (0.0, 0.1, 0.1)
           12669467083591725_u64,   // (0.1, 0.0, 0.1)
           17737253917028415_u64,   // (0.1, 0.1, 0.1)
           141898031336227320_u64,  // (0.2, 0.2, 0.2)
           1011023473270619655_u64, // (0.3, 0.3, 0.3)
           1135184250689818560_u64, // (0.4, 0.4, 0.4)
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64};

    std::vector<Tmorton> test_mortons_sorted
        = {0_u64,                   // (0.0, 0.0, 0.0)
           2533893416718345_u64,    // (0.0, 0.0, 0.1)
           5067786833436690_u64,    // (0.0, 0.1, 0.0)
           7601680250155035_u64,    // (0.0, 0.1, 0.1)
           10135573666873380_u64,   // (0.1, 0.0, 0.0)
           12669467083591725_u64,   // (0.1, 0.0, 0.1)
           15203360500310070_u64,   // (0.1, 0.1, 0.0)
           17737253917028415_u64,   // (0.1, 0.1, 0.1)
           141898031336227320_u64,  // (0.2, 0.2, 0.2)
           1011023473270619655_u64, // (0.3, 0.3, 0.3)
           1135184250689818560_u64, // (0.4, 0.4, 0.4)
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64};

    std::vector<u32> index_map_obj_idx = {0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 15, 11, 14, 12, 13};

    std::vector<Tmorton> reduced_morton_codes = {
        0_u64,                  // (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)
        5067786833436690_u64,   // (0.0, 0.1, 0.0), (0.0, 0.1, 0.1)
        10135573666873380_u64,  // (0.1, 0.0, 0.0), (0.1, 0.0, 0.1)
        15203360500310070_u64,  // (0.1, 0.1, 0.0), (0.1, 0.1, 0.1)
        141898031336227320_u64, // (0.2, 0.2, 0.2)
        1011023473270619655_u64 // (0.3, 0.3, 0.3),(0.4, 0.4, 0.4)
    };

    std::vector<u32> buf_reduc_index_map = {0, 2, 4, 6, 8, 9, 11, 0};

    std::vector<u32> expected_lchild_id  = {4, 0, 2, 1, 3};
    std::vector<u8> expected_lchild_flag = {0, 1, 1, 0, 0};
    std::vector<u32> expected_rchild_id  = {5, 1, 3, 2, 4};
    std::vector<u8> expected_rchild_flag = {1, 1, 1, 0, 1};
    std::vector<u32> expected_endrange   = {5, 0, 3, 0, 0};

    /*
    l0 = (0.0, 0.0, 0.0)x(0.0, 0.0, 0.1)
    l1 = (0.0, 0.1, 0.0)x(0.0, 0.1, 0.1)
    l2 = (0.1, 0.0, 0.0)x(0.1, 0.0, 0.1)
    l3 = (0.1, 0.1, 0.0)x(0.1, 0.1, 0.1)
    l4 = (0.2, 0.2, 0.2)x(0.2, 0.2, 0.2)
    l5 = (0.3, 0.3, 0.3)x(0.4, 0.4, 0.4)

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

    i1 = (0.0, 0.0, 0.0)x(0.0, 0.1, 0.1)
    i2 = (0.1, 0.0, 0.0)x(0.1, 0.1, 0.1)
    i3 = (0.0, 0.0, 0.0)x(0.1, 0.1, 0.1)
    i4 = (0.0, 0.0, 0.0)x(0.2, 0.2, 0.2)
    i0 = (0.0, 0.0, 0.0)x(0.4, 0.4, 0.4)
    */

    std::vector<Tvec> aabb_min = {
        {0.0, 0.0, 0.0}, // i0
        {0.0, 0.0, 0.0}, // i1
        {0.1, 0.0, 0.0}, // i2
        {0.0, 0.0, 0.0}, // i3
        {0.0, 0.0, 0.0}, // i4
        {0.0, 0.0, 0.0}, // l0
        {0.0, 0.1, 0.0}, // l1
        {0.1, 0.0, 0.0}, // l2
        {0.1, 0.1, 0.0}, // l3
        {0.2, 0.2, 0.2}, // l4
        {0.3, 0.3, 0.3}  // l5
    };
    std::vector<Tvec> aabb_max = {
        {0.4, 0.4, 0.4}, // i0
        {0.0, 0.1, 0.1}, // i1
        {0.1, 0.1, 0.1}, // i2
        {0.1, 0.1, 0.1}, // i3
        {0.2, 0.2, 0.2}, // i4
        {0.0, 0.0, 0.1}, // l0
        {0.0, 0.1, 0.1}, // l1
        {0.1, 0.0, 0.1}, // l2
        {0.1, 0.1, 0.1}, // l3
        {0.2, 0.2, 0.2}, // l4
        {0.4, 0.4, 0.4}  // l5
    };

    /// Run of the test

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(
        partpos.size(), shamsys::instance::get_compute_scheduler_ptr());

    partpos_buf.copy_from_stdvec(partpos);

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, 3>(
        shamsys::instance::get_compute_scheduler_ptr(), bb, partpos_buf, partpos.size(), 16);

    std::vector<Tmorton> mortons = set.morton_codes.copy_to_stdvec();

    REQUIRE_EQUAL(set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(set.morton_count, 16);
    REQUIRE_EQUAL(set.morton_codes.get_size(), 16);
    REQUIRE_EQUAL(set.morton_codes.copy_to_stdvec(), test_mortons);

    auto sorted_set
        = shamtree::sort_morton_set(shamsys::instance::get_compute_scheduler_ptr(), std::move(set));

    REQUIRE_EQUAL(sorted_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(sorted_set.morton_count, 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);

    auto reduced_set = shamtree::reduce_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(), std::move(sorted_set), 1);

    REQUIRE_EQUAL(reduced_set.morton_codes_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(reduced_set.morton_codes_set.morton_count, 16);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);

    REQUIRE_EQUAL(reduced_set.reduce_code_count, 6);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.get_size(), 6);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.copy_to_stdvec(), reduced_morton_codes);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.get_size(), 6 + 2);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.copy_to_stdvec(), buf_reduc_index_map);

    auto tree = shamtree::karras_tree_from_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(),
        reduced_set.reduced_morton_codes.get_size(),
        reduced_set.reduced_morton_codes);

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

    auto aabbs = shamtree::compute_tree_aabb_from_positions(
        tree,
        reduced_set.get_cell_iterator(),
        shamtree::new_empty_karras_radix_tree_aabb<Tvec>(),
        partpos_buf);

    auto ret_aabb_min = aabbs.buf_aabb_min.copy_to_stdvec();
    auto ret_aabb_max = aabbs.buf_aabb_max.copy_to_stdvec();

    REQUIRE_EQUAL(aabbs.buf_aabb_min.get_size(), 2 * 5 + 1);
    REQUIRE_EQUAL(aabbs.buf_aabb_max.get_size(), 2 * 5 + 1);

    auto vec_equals = [](std::vector<Tvec> a, std::vector<Tvec> b) {
        bool same_size = a.size() == b.size();
        if (!same_size) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            same_size = same_size && sham::equals(a[i], b[i]);
        }
        return same_size;
    };

    REQUIRE_EQUAL_CUSTOM_COMP(ret_aabb_min, aabb_min, vec_equals);
    REQUIRE_EQUAL_CUSTOM_COMP(ret_aabb_max, aabb_max, vec_equals);
}

TestStart(
    Unittest, "shamtree/KarrasRadixTreeAABB(one-cell)", test_karras_radix_tree_aabb_one_cell, 1) {

    std::vector<Tvec> partpos{
        Tvec(0.0, 0.0, 0.0),
        Tvec(0.0, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.0),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4)};

    std::vector<Tmorton> test_mortons
        = {0_u64,
           2533893416718345_u64,
           15203360500310070_u64,
           1011023473270619655_u64,
           1135184250689818560_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64};

    std::vector<Tmorton> test_mortons_sorted
        = {0_u64,
           2533893416718345_u64,
           15203360500310070_u64,
           1011023473270619655_u64,
           1135184250689818560_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64,
           18446744073709551615_u64};

    std::vector<u32> index_map_obj_idx = {0, 1, 2, 3, 4, 15, 12, 14, 6, 5, 7, 13, 9, 8, 10, 11};

    std::vector<Tmorton> reduced_morton_codes = {
        0_u64,
    };

    std::vector<u32> buf_reduc_index_map = {0, 5, 0};

    std::vector<u32> expected_lchild_id  = {};
    std::vector<u8> expected_lchild_flag = {};
    std::vector<u32> expected_rchild_id  = {};
    std::vector<u8> expected_rchild_flag = {};
    std::vector<u32> expected_endrange   = {};

    std::vector<Tvec> aabb_min = {
        {0.0, 0.0, 0.0},
    };
    std::vector<Tvec> aabb_max = {
        {0.4, 0.4, 0.4},
    };

    /// Run of the test

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(
        partpos.size(), shamsys::instance::get_compute_scheduler_ptr());

    partpos_buf.copy_from_stdvec(partpos);

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, 3>(
        shamsys::instance::get_compute_scheduler_ptr(), bb, partpos_buf, partpos.size(), 16);

    std::vector<Tmorton> mortons = set.morton_codes.copy_to_stdvec();

    REQUIRE_EQUAL(set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(set.morton_count, 16);
    REQUIRE_EQUAL(set.morton_codes.get_size(), 16);
    REQUIRE_EQUAL(set.morton_codes.copy_to_stdvec(), test_mortons);

    auto sorted_set
        = shamtree::sort_morton_set(shamsys::instance::get_compute_scheduler_ptr(), std::move(set));

    REQUIRE_EQUAL(sorted_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(sorted_set.morton_count, 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);

    auto reduced_set = shamtree::reduce_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(), std::move(sorted_set), 5);

    REQUIRE_EQUAL(reduced_set.morton_codes_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(reduced_set.morton_codes_set.morton_count, 16);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);

    REQUIRE_EQUAL(reduced_set.reduce_code_count, 1);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.get_size(), 1);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.copy_to_stdvec(), reduced_morton_codes);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.get_size(), 1 + 2);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.copy_to_stdvec(), buf_reduc_index_map);

    auto tree = shamtree::karras_tree_from_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(),
        reduced_set.reduced_morton_codes.get_size(),
        reduced_set.reduced_morton_codes);

    REQUIRE_EQUAL(tree.buf_lchild_id.copy_to_stdvec(), expected_lchild_id);
    REQUIRE_EQUAL(tree.buf_rchild_id.copy_to_stdvec(), expected_rchild_id);
    REQUIRE_EQUAL(tree.buf_lchild_flag.copy_to_stdvec(), expected_lchild_flag);
    REQUIRE_EQUAL(tree.buf_rchild_flag.copy_to_stdvec(), expected_rchild_flag);
    REQUIRE_EQUAL(tree.buf_endrange.copy_to_stdvec(), expected_endrange);

    REQUIRE_EQUAL(tree.buf_lchild_id.get_size(), 0);
    REQUIRE_EQUAL(tree.buf_rchild_id.get_size(), 0);
    REQUIRE_EQUAL(tree.buf_lchild_flag.get_size(), 0);
    REQUIRE_EQUAL(tree.buf_rchild_flag.get_size(), 0);
    REQUIRE_EQUAL(tree.buf_endrange.get_size(), 0);

    auto aabbs = shamtree::compute_tree_aabb_from_positions(
        tree,
        reduced_set.get_cell_iterator(),
        shamtree::new_empty_karras_radix_tree_aabb<Tvec>(),
        partpos_buf);

    auto ret_aabb_min = aabbs.buf_aabb_min.copy_to_stdvec();
    auto ret_aabb_max = aabbs.buf_aabb_max.copy_to_stdvec();

    REQUIRE_EQUAL(aabbs.buf_aabb_min.get_size(), 2 * 0 + 1);
    REQUIRE_EQUAL(aabbs.buf_aabb_max.get_size(), 2 * 0 + 1);

    auto vec_equals = [](std::vector<Tvec> a, std::vector<Tvec> b) {
        bool same_size = a.size() == b.size();
        if (!same_size) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            same_size = same_size && sham::equals(a[i], b[i]);
        }
        return same_size;
    };

    REQUIRE_EQUAL_CUSTOM_COMP(ret_aabb_min, aabb_min, vec_equals);
    REQUIRE_EQUAL_CUSTOM_COMP(ret_aabb_max, aabb_max, vec_equals);
}

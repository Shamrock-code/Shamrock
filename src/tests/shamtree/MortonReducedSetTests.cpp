// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/integer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamtree/MortonReducedSet.hpp"
#include <vector>

using Tvec    = f64_3;
using Tmorton = u64;

TestStart(Unittest, "shamtree/MortonReducedSet", test_morton_reduced_set, 1) {

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

    std::vector<Tmorton> test_mortons
        = {0b0000000000000000000000000000000000000000000000000000000000000000,
           0b0000000000100100000000100100000000100100000000100100000000100100,
           0b0000000000010010000000010010000000010010000000010010000000010010,
           0b0000000000001001000000001001000000001001000000001001000000001001,
           0b0000000000110110000000110110000000110110000000110110000000110110,
           0b0000000000011011000000011011000000011011000000011011000000011011,
           0b0000000000101101000000101101000000101101000000101101000000101101,
           0b0000000000111111000000111111000000111111000000111111000000111111,
           0b0000000111111000000111111000000111111000000111111000000111111000,
           0b0000111000000111111000000111111000000111111000000111111000000111,
           0b0000111111000000111111000000111111000000111111000000111111000000,
           0b0111111111111111111111111111111111111111111111111111111111111111,
           0b0111111111111111111111111111111111111111111111111111111111111111,
           0b0000000000000000000000000000000000000000000000000000000000000000,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111};

    std::vector<Tmorton> test_mortons_sorted
        = {0b0000000000000000000000000000000000000000000000000000000000000000,  // 0  -> 0
           0b0000000000000000000000000000000000000000000000000000000000000000,  // 1  -> 0
           0b0000000000001001000000001001000000001001000000001001000000001001,  // 2  -> 1
           0b0000000000010010000000010010000000010010000000010010000000010010,  // 3  -> 2
           0b0000000000011011000000011011000000011011000000011011000000011011,  // 4  -> 3
           0b0000000000100100000000100100000000100100000000100100000000100100,  // 5  -> 4
           0b0000000000101101000000101101000000101101000000101101000000101101,  // 6  -> 5
           0b0000000000110110000000110110000000110110000000110110000000110110,  // 7  -> 6
           0b0000000000111111000000111111000000111111000000111111000000111111,  // 8  -> 7
           0b0000000111111000000111111000000111111000000111111000000111111000,  // 9  -> 8
           0b0000111000000111111000000111111000000111111000000111111000000111,  // 10 -> 9
           0b0000111111000000111111000000111111000000111111000000111111000000,  // 11 -> 10
           0b0111111111111111111111111111111111111111111111111111111111111111,  // 12 -> 11
           0b0111111111111111111111111111111111111111111111111111111111111111,  // 13 -> 11
           0b1111111111111111111111111111111111111111111111111111111111111111,  // 14 -> na
           0b1111111111111111111111111111111111111111111111111111111111111111}; // 15 -> na

    // karras result
    // int cell (0) ----------------------------------------------------------------------        //
    // int cell ------------------------------------------------------------- (10) \              //
    // int cell ------------------------------------------------ (8) (9) -------    \             //
    // int cell ------------------------------------------ (7)    |   |     \        \            //
    // int cell --------------- (3) (4) --------------------      |   |      \        \           //
    // int cell --- (1)   (2) ---    ------- (5) (6) -------      |   |       \        \          //
    //            /  |     |   \        /     |   |        \      |   |        \        \         //
    // leafs    (0) (1)   (2)   (3) (4)      (5) (6)       (7)   (8) (9)      (10)     (11)       //

    // reduc delete second code if collapse ok
    // Expected reduction 1
    // 0 2 4 6 8 9 11

    // Expected reduction 2
    // 0 4 8 9 11

    // result indexes
    // 0 5 9 10 12

    std::vector<u32> index_map_obj_idx = {13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, 15, 14};

    std::vector<Tmorton> reduced_morton_codes
        = {0b0000000000000000000000000000000000000000000000000000000000000000,
           0b0000000000100100000000100100000000100100000000100100000000100100,
           0b0000000111111000000111111000000111111000000111111000000111111000,
           0b0000111000000111111000000111111000000111111000000111111000000111,
           0b0111111111111111111111111111111111111111111111111111111111111111};

    std::vector<u32> buf_reduc_index_map = {0, 5, 9, 10, 12, 14, 0};

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
        shamsys::instance::get_compute_scheduler_ptr(), std::move(sorted_set), 2);

    REQUIRE_EQUAL(reduced_set.morton_codes_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(reduced_set.morton_codes_set.morton_count, 16);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(reduced_set.morton_codes_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(
        reduced_set.morton_codes_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);

    REQUIRE_EQUAL(reduced_set.reduce_code_count, 5);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.get_size(), 5);
    REQUIRE_EQUAL(reduced_set.reduced_morton_codes.copy_to_stdvec(), reduced_morton_codes);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.get_size(), 5 + 2);
    REQUIRE_EQUAL(reduced_set.buf_reduc_index_map.copy_to_stdvec(), buf_reduc_index_map);
}

TestStart(Unittest, "shamtree/MortonReducedSet(single cell)", test_morton_reduced_set_onecell, 1) {

    std::vector<Tvec> partpos{Tvec(0.1, 0.0, 0.0), Tvec(0.0, 0.1, 0.0), Tvec(0.0, 0.0, 0.1)};

    std::vector<Tmorton> test_mortons
        = {0b0000000000100100000000100100000000100100000000100100000000100100,
           0b0000000000010010000000010010000000010010000000010010000000010010,
           0b0000000000001001000000001001000000001001000000001001000000001001,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111};

    std::vector<Tmorton> test_mortons_sorted
        = {0b0000000000001001000000001001000000001001000000001001000000001001,
           0b0000000000010010000000010010000000010010000000010010000000010010,
           0b0000000000100100000000100100000000100100000000100100000000100100,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111,
           0b1111111111111111111111111111111111111111111111111111111111111111};

    std::vector<u32> index_map_obj_idx = {2, 1, 0, 15, 9, 14, 12, 13, 5, 4, 6, 3, 7, 8, 10, 11};

    std::vector<Tmorton> reduced_morton_codes
        = {0b0000000000001001000000001001000000001001000000001001000000001001};

    std::vector<u32> buf_reduc_index_map = {0, 3, 0};

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
        shamsys::instance::get_compute_scheduler_ptr(), std::move(sorted_set), 2);

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
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include <vector>

using Tmorton = u64;

TestStart(Unittest, "shamtree/KarrasRadixTree", test_karras_radix_tree, 1) {

    std::vector<Tmorton> test_morton_codes = {
        0b0000000000000000000000000000000000000000000000000000000000000000, // 0
        0b0000000000001001000000001001000000001001000000001001000000001001, // 1
        0b0000000000010010000000010010000000010010000000010010000000010010, // 2
        0b0000000000011011000000011011000000011011000000011011000000011011, // 3
        0b0000000000100100000000100100000000100100000000100100000000100100, // 4
        0b0000000000101101000000101101000000101101000000101101000000101101, // 5
        0b0000000000110110000000110110000000110110000000110110000000110110, // 6
        0b0000000000111111000000111111000000111111000000111111000000111111, // 7
        0b0000000111111000000111111000000111111000000111111000000111111000, // 8
        0b0000111000000111111000000111111000000111111000000111111000000111, // 9
        0b0000111111000000111111000000111111000000111111000000111111000000, // 10
        0b0111111111111111111111111111111111111111111111111111111111111111, // 11
    };

    // karras result
    // int cell (0) ----------------------------------------------------------------------        //
    // int cell ------------------------------------------------------------- (10) \              //
    // int cell ------------------------------------------------ (8) (9) -------    \             //
    // int cell ------------------------------------------ (7)    |   |     \        \            //
    // int cell --------------- (3) (4) --------------------      |   |      \        \           //
    // int cell --- (1)   (2) ---    ------- (5) (6) -------      |   |       \        \          //
    //            /  |     |   \        /     |   |        \      |   |        \        \         //
    // leafs    (0) (1)   (2)   (3) (4)      (5) (6)       (7)   (8) (9)      (10)     (11)       //

    std::vector<u32> expected_lchild_id  = {10, 0, 2, 1, 5, 4, 6, 3, 7, 9, 8};
    std::vector<u8> expected_lchild_flag = {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    std::vector<u32> expected_rchild_id  = {11, 1, 3, 2, 6, 5, 7, 4, 8, 10, 9};
    std::vector<u8> expected_rchild_flag = {1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0};
    std::vector<u32> expected_endrange   = {11, 0, 3, 0, 7, 4, 7, 0, 0, 10, 0};

    sham::DeviceBuffer<Tmorton> morton_codes(
        test_morton_codes.size(), shamsys::instance::get_compute_scheduler_ptr());

    morton_codes.copy_from_stdvec(test_morton_codes);

    auto tree = shamtree::karras_tree_from_reduced_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(), test_morton_codes.size(), morton_codes);

    REQUIRE_EQUAL(tree.buf_lchild_id.copy_to_stdvec(), expected_lchild_id);
    REQUIRE_EQUAL(tree.buf_rchild_id.copy_to_stdvec(), expected_rchild_id);
    REQUIRE_EQUAL(tree.buf_lchild_flag.copy_to_stdvec(), expected_lchild_flag);
    REQUIRE_EQUAL(tree.buf_rchild_flag.copy_to_stdvec(), expected_rchild_flag);
    REQUIRE_EQUAL(tree.buf_endrange.copy_to_stdvec(), expected_endrange);

    REQUIRE_EQUAL(tree.buf_lchild_id.get_size(), 11);
    REQUIRE_EQUAL(tree.buf_rchild_id.get_size(), 11);
    REQUIRE_EQUAL(tree.buf_lchild_flag.get_size(), 11);
    REQUIRE_EQUAL(tree.buf_rchild_flag.get_size(), 11);
    REQUIRE_EQUAL(tree.buf_endrange.get_size(), 11);
}

TestStart(Unittest, "shamtree/KarrasRadixTree(one-cell)", test_karras_radix_tree_one_cell, 1) {

    // In this test we supply only a single morton code, as such the tree is just a single leaf.
    // As such every buffer should be empty as there is no tree structure.

    std::vector<Tmorton> test_morton_codes = {
        0b0000000000100100000000100100000000100100000000100100000000100100,
    };

    std::vector<u32> expected_lchild_id  = {};
    std::vector<u8> expected_lchild_flag = {};
    std::vector<u32> expected_rchild_id  = {};
    std::vector<u8> expected_rchild_flag = {};
    std::vector<u32> expected_endrange   = {};

    sham::DeviceBuffer<Tmorton> morton_codes(
        test_morton_codes.size(), shamsys::instance::get_compute_scheduler_ptr());

    morton_codes.copy_from_stdvec(test_morton_codes);

    auto tree = shamtree::karras_tree_from_reduced_morton_set(
        shamsys::instance::get_compute_scheduler_ptr(), test_morton_codes.size(), morton_codes);

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
}

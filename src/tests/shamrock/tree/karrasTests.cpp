// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

// #include "test_tree.hpp"

#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/kernels/karras_alg.hpp"
#include <memory>
#include <vector>

template<class mprec>
inline void test_karras_alg() {
    using u_morton = mprec;

    std::vector<u_morton> morton_list = {
        0x0,
        0x1,
        0x2,
        0x3,
        0x4,
        0x5,
        0x6,
        0x7,
        0x8,
        // 0x9,
        // 0xa,
        // 0xb,
        0xc,
        0xd,
        0xe,
        0xf,
    };

    std::vector<u32> out_lchild_id(morton_list.size());
    std::vector<u32> out_rchild_id(morton_list.size());
    std::vector<u8> out_lchild_flag(morton_list.size());
    std::vector<u8> out_rchild_flag(morton_list.size());
    std::vector<u32> out_endrange(morton_list.size());

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton
            = std::make_unique<sycl::buffer<u_morton>>(morton_list.data(), morton_list.size());
        std::unique_ptr<sycl::buffer<u32>> out_buf_lchild_id
            = std::make_unique<sycl::buffer<u32>>(out_lchild_id.data(), out_lchild_id.size());
        std::unique_ptr<sycl::buffer<u32>> out_buf_rchild_id
            = std::make_unique<sycl::buffer<u32>>(out_rchild_id.data(), out_rchild_id.size());
        std::unique_ptr<sycl::buffer<u8>> out_buf_lchild_flag
            = std::make_unique<sycl::buffer<u8>>(out_lchild_flag.data(), out_lchild_flag.size());
        std::unique_ptr<sycl::buffer<u8>> out_buf_rchild_flag
            = std::make_unique<sycl::buffer<u8>>(out_rchild_flag.data(), out_rchild_flag.size());
        std::unique_ptr<sycl::buffer<u32>> out_buf_endrange
            = std::make_unique<sycl::buffer<u32>>(out_endrange.data(), out_endrange.size());

        sycl_karras_alg<u_morton>(
            shamsys::instance::get_compute_queue(),
            morton_list.size() - 1,
            *buf_morton,
            *out_buf_lchild_id,
            *out_buf_rchild_id,
            *out_buf_lchild_flag,
            *out_buf_rchild_flag,
            *out_buf_endrange);
    }

    REQUIRE_EQUAL(out_lchild_id[0], 7);
    REQUIRE_EQUAL(out_lchild_id[1], 0);
    REQUIRE_EQUAL(out_lchild_id[2], 2);
    REQUIRE_EQUAL(out_lchild_id[3], 1);
    REQUIRE_EQUAL(out_lchild_id[4], 5);
    REQUIRE_EQUAL(out_lchild_id[5], 4);
    REQUIRE_EQUAL(out_lchild_id[6], 6);
    REQUIRE_EQUAL(out_lchild_id[7], 3);
    REQUIRE_EQUAL(out_lchild_id[8], 8);
    REQUIRE_EQUAL(out_lchild_id[9], 10);
    REQUIRE_EQUAL(out_lchild_id[10], 9);
    REQUIRE_EQUAL(out_lchild_id[11], 11);

    REQUIRE_EQUAL(out_rchild_id[0], 8);
    REQUIRE_EQUAL(out_rchild_id[1], 1);
    REQUIRE_EQUAL(out_rchild_id[2], 3);
    REQUIRE_EQUAL(out_rchild_id[3], 2);
    REQUIRE_EQUAL(out_rchild_id[4], 6);
    REQUIRE_EQUAL(out_rchild_id[5], 5);
    REQUIRE_EQUAL(out_rchild_id[6], 7);
    REQUIRE_EQUAL(out_rchild_id[7], 4);
    REQUIRE_EQUAL(out_rchild_id[8], 9);
    REQUIRE_EQUAL(out_rchild_id[9], 11);
    REQUIRE_EQUAL(out_rchild_id[10], 10);
    REQUIRE_EQUAL(out_rchild_id[11], 12);

    REQUIRE_EQUAL(out_lchild_flag[0], 0);
    REQUIRE_EQUAL(out_lchild_flag[1], 1);
    REQUIRE_EQUAL(out_lchild_flag[2], 1);
    REQUIRE_EQUAL(out_lchild_flag[3], 0);
    REQUIRE_EQUAL(out_lchild_flag[4], 0);
    REQUIRE_EQUAL(out_lchild_flag[5], 1);
    REQUIRE_EQUAL(out_lchild_flag[6], 1);
    REQUIRE_EQUAL(out_lchild_flag[7], 0);
    REQUIRE_EQUAL(out_lchild_flag[8], 1);
    REQUIRE_EQUAL(out_lchild_flag[9], 0);
    REQUIRE_EQUAL(out_lchild_flag[10], 1);
    REQUIRE_EQUAL(out_lchild_flag[11], 1);

    REQUIRE_EQUAL(out_rchild_flag[0], 0);
    REQUIRE_EQUAL(out_rchild_flag[1], 1);
    REQUIRE_EQUAL(out_rchild_flag[2], 1);
    REQUIRE_EQUAL(out_rchild_flag[3], 0);
    REQUIRE_EQUAL(out_rchild_flag[4], 0);
    REQUIRE_EQUAL(out_rchild_flag[5], 1);
    REQUIRE_EQUAL(out_rchild_flag[6], 1);
    REQUIRE_EQUAL(out_rchild_flag[7], 0);
    REQUIRE_EQUAL(out_rchild_flag[8], 0);
    REQUIRE_EQUAL(out_rchild_flag[9], 0);
    REQUIRE_EQUAL(out_rchild_flag[10], 1);
    REQUIRE_EQUAL(out_rchild_flag[11], 1);

    REQUIRE_EQUAL(out_endrange[0], 12);
    REQUIRE_EQUAL(out_endrange[1], 0);
    REQUIRE_EQUAL(out_endrange[2], 3);
    REQUIRE_EQUAL(out_endrange[3], 0);
    REQUIRE_EQUAL(out_endrange[4], 7);
    REQUIRE_EQUAL(out_endrange[5], 4);
    REQUIRE_EQUAL(out_endrange[6], 7);
    REQUIRE_EQUAL(out_endrange[7], 0);
    REQUIRE_EQUAL(out_endrange[8], 12);
    REQUIRE_EQUAL(out_endrange[9], 12);
    REQUIRE_EQUAL(out_endrange[10], 9);
    REQUIRE_EQUAL(out_endrange[11], 12);
}

TestStart(Unittest, "core/tree/kernels/karras_alg (32)", karras_utest32, 1) {
    test_karras_alg<u32>();
}

TestStart(Unittest, "core/tree/kernels/karras_alg (64)", karras_utest64, 1) {
    test_karras_alg<u64>();
}

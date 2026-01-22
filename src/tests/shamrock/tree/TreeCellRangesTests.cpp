// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shammath/CoordRange.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/TreeCellRanges.hpp"
#include "shamtree/TreeMortonCodes.hpp"
#include "shamtree/TreeReducedMortonCodes.hpp"
#include "shamtree/TreeStructure.hpp"

TestStart(Unittest, "shamrock/tree/TreeCellRanges::serialize", testcellrangesserialize, 1) {

    using pos_t    = f32_3;
    using u_morton = u32;

    u32 cnt = 1000;
    shammath::CoordRange<pos_t> range_coord{pos_t{0, 0, 0}, pos_t{1, 1, 1}};

    sycl::queue &q = shamsys::instance::get_compute_queue();

    auto buf = shamalgs::random::mock_buffer(0x1111, cnt, range_coord.lower, range_coord.upper);

    using TreeMorton    = shamrock::tree::TreeMortonCodes<u_morton>;
    using TreeRedMorton = shamrock::tree::TreeReducedMortonCodes<u_morton>;
    using TreeStruct    = shamrock::tree::TreeStructure<u_morton>;
    using TreeRanges    = shamrock::tree::TreeCellRanges<u_morton, pos_t>;

    TreeMorton codes;
    codes.build(q, range_coord, cnt, buf);

    bool one_cell_mode;
    TreeRedMorton redcodes;
    redcodes.build(q, codes.obj_cnt, 2, codes, one_cell_mode);

    TreeStruct strc;
    if (!one_cell_mode) {
        strc.build(q, redcodes.tree_leaf_count - 1, *redcodes.buf_tree_morton);
    } else {
        strc.build_one_cell_mode();
    }

    TreeRanges rnge;
    rnge.build1(q, redcodes, strc);
    rnge.build2(
        q,
        redcodes.tree_leaf_count + strc.internal_cell_count,
        {range_coord.lower, range_coord.upper});

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
    ser.allocate(rnge.serialize_byte_size());
    rnge.serialize(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        TreeRanges outser = TreeRanges::deserialize(ser2);

        REQUIRE_NAMED("input match out", outser == rnge);
    }
}

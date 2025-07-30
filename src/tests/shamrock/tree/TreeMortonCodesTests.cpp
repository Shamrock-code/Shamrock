// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/TreeMortonCodes.hpp"

TestStart(
    Unittest, "shamrock/tree/TreeMortonCodes::serialize", test_tree_morton_code_serialize, 1) {

    u32 cnt = 1000;
    shammath::CoordRange<f32_3> range_coord{f32_3{0, 0, 0}, f32_3{1, 1, 1}};

    sycl::queue &q = shamsys::instance::get_compute_queue();

    auto buf = shamalgs::random::mock_buffer(0x1111, cnt, range_coord.lower, range_coord.upper);

    using u_morton = u32;

    using TreeMorton = shamrock::tree::TreeMortonCodes<u_morton>;

    TreeMorton codes;
    codes.build(q, range_coord, cnt, buf);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
    ser.allocate(codes.serialize_byte_size());
    codes.serialize(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        TreeMorton outser = TreeMorton::deserialize(ser2);

        REQUIRE_NAMED("input match out", codes == outser);
    }
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamrock/tree/TreeMortonCodes.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

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

        shamtest::asserts().assert_bool("input match out", codes == outser);
    }
}

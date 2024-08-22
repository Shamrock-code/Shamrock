// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/container/ResizableBuffer.hpp"
#include "shamalgs/serialize.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/container/ResizableBuffer", resizebuftestserialize, 1) {

    u32 len = 1000;

    using namespace shamalgs;

    ResizableBuffer<u32_3> buf = ResizableBuffer<u32_3>::mock_buffer(
        shamsys::instance::get_compute_scheduler_ptr(),
        0x111,
        len,
        u32_3{0, 0, 0},
        u32_3{100, 100, 100});

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    ser.allocate(buf.serialize_buf_byte_size());
    buf.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        ResizableBuffer<u32_3> buf2 = ResizableBuffer<u32_3>::deserialize_buf(ser2, len);

        shamtest::asserts().assert_bool("input match out", buf.check_buf_match(buf2));
    }
}

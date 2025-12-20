// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(
    Unittest, "shamsys/comm/CommunicationBuffer/constructor", test_basic_serialized_constr, 1) {

    u32 nbytes                      = 1e5;
    sham::DeviceBuffer<u8> buf_comp = shamalgs::random::mock_buffer_usm<u8>(
        shamsys::instance::get_compute_scheduler_ptr(), 0x111, nbytes);

    {
        shamcomm::CommunicationBuffer cbuf{
            buf_comp, shamsys::instance::get_compute_scheduler_ptr()};
        sham::DeviceBuffer<u8> ret = shamcomm::CommunicationBuffer::convert_usm(std::move(cbuf));
        REQUIRE_EQUAL(buf_comp.copy_to_stdvec(), ret.copy_to_stdvec());
    }

}

TestStart(
    Unittest, "shamsys/comm/CommunicationBuffer/send_recv", test_basic_serialized_send_recv, 2) {

    u32 nbytes                      = 1e5;
    sham::DeviceBuffer<u8> buf_comp = shamalgs::random::mock_buffer_usm<u8>(
        shamsys::instance::get_compute_scheduler_ptr(), 0x111, nbytes);

    if (shamcomm::world_rank() == 0) {
        shamcomm::CommunicationBuffer cbuf{
            buf_comp, shamsys::instance::get_compute_scheduler_ptr()};
        MPI_Send(cbuf.get_ptr(), nbytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    }

    if (shamcomm::world_rank() == 1) {
        shamcomm::CommunicationBuffer cbuf{nbytes, shamsys::instance::get_compute_scheduler_ptr()};
        MPI_Recv(cbuf.get_ptr(), nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        sham::DeviceBuffer<u8> ret = shamcomm::CommunicationBuffer::convert_usm(std::move(cbuf));
        REQUIRE_EQUAL(buf_comp.copy_to_stdvec(), ret.copy_to_stdvec());
    }
}

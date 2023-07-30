// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shambase/DistributedData.hpp"
#include "shamsys/comm/details/CommunicationBufferImpl.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <map>
#include <memory>


void distribdata_sparse_comm_test(std::string prefix, shamsys::CommunicationProtocol prot){

    using namespace shamalgs::collective;
    using namespace shamsys::instance;
    using namespace shamsys;

    const i32 wsize = world_size;
    const i32 wrank = world_rank;

    u32 npatch       = wsize * 5;
    u32 nbuf_p_patch = 2;
    u64 max_msg_len  = 1e4;

    u64 seed = 0x123;

    std::mt19937 eng(seed);

    std::map<u64, i32> rank_owner;
    for (u64 i = 0; i < npatch; i++) {
        rank_owner[i] = shamalgs::random::mock_value(eng, 0, wsize - 1);
    }

    shamalgs::collective::SerializedDDataComm dat_ref;

    for (u64 i = 0; i < npatch * nbuf_p_patch; i++) {
        u64 sender   = shamalgs::random::mock_value(eng, 0_u64, npatch - 1_u64);
        u64 receiver = shamalgs::random::mock_value(eng, 0_u64, npatch - 1_u64);
        u64 lenght   = shamalgs::random::mock_value(eng, 1_u64, max_msg_len);
        u64 rnd      = eng();

        if (!dat_ref.has_key(sender, receiver)) {
            dat_ref.add_obj(sender,
                            receiver,
                            std::make_unique<sycl::buffer<u8>>(shamalgs::random::mock_buffer<u8>(
                                rnd, shamalgs::random::mock_value<i32>(eng, 1, lenght))));
        }
    }

    shamalgs::collective::SerializedDDataComm send_data;

    dat_ref.for_each([&](u64 sender, u64 receiver, std::unique_ptr<sycl::buffer<u8>> &buf) {
        if (rank_owner[sender] == wrank) {
            send_data.add_obj(sender, receiver, shamalgs::memory::duplicate(buf));
        }
    });

    shamalgs::collective::SerializedDDataComm recv_data;
    distributed_data_sparse_comm(
        send_data, recv_data, prot, [&](u64 id) { return rank_owner[id]; });

    shamalgs::collective::SerializedDDataComm recv_data_ref;
    dat_ref.for_each([&](u64 sender, u64 receiver, std::unique_ptr<sycl::buffer<u8>> &buf) {
        if (rank_owner[receiver] == wrank) {
            recv_data_ref.add_obj(sender, receiver, shamalgs::memory::duplicate(buf));
        }
    });

    shamtest::asserts().assert_equal("expected number of recv",
                                     recv_data.get_element_count(),
                                     recv_data_ref.get_element_count());

    recv_data_ref.for_each([&](u64 sender, u64 receiver, std::unique_ptr<sycl::buffer<u8>> &buf) {
        shamtest::asserts().assert_bool("has expected key", recv_data.has_key(sender, receiver));

        auto it = recv_data.get_native().find({sender, receiver});

        shamtest::asserts().assert_bool(
            "correct buffer",
            shamalgs::reduction::equals_ptr(buf, it->second));
    });
}

TestStart(Unittest, "shamalgs/collective/distributedDataComm", testdistributeddatacomm, -1) {

    distribdata_sparse_comm_test("CopyToHost mode : ",shamsys::CopyToHost);
    distribdata_sparse_comm_test("DirectGPU  mode : ",shamsys::DirectGPU);
}
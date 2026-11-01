// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <vector>

void distribdata_sparse_comm_test(
    u32 npatch,
    u32 nbuf_p_patch,
    u64 max_msg_len,
    u64 seed,
    std::string prefix,
    u64 max_comm_size) {

    using namespace shamalgs::collective;
    using namespace shamsys::instance;
    using namespace shamsys;

    const i32 wsize = shamcomm::world_size();
    const i32 wrank = shamcomm::world_rank();

    std::mt19937 eng(seed);

    std::map<u64, i32> rank_owner;
    for (u64 i = 0; i < npatch; i++) {
        rank_owner[i] = shamalgs::primitives::mock_value(eng, 0, wsize - 1);
    }

    shamalgs::collective::SerializedDDataComm dat_ref;

    for (u64 i = 0; i < npatch * nbuf_p_patch; i++) {
        u64 sender   = shamalgs::primitives::mock_value(eng, 0_u64, npatch - 1_u64);
        u64 receiver = shamalgs::primitives::mock_value(eng, 0_u64, npatch - 1_u64);
        u64 length   = shamalgs::primitives::mock_value(eng, 1_u64, max_msg_len);
        u64 rnd      = eng();

        if (!dat_ref.has_key(sender, receiver)) {
            dat_ref.add_obj(
                sender,
                receiver,
                shamalgs::random::mock_buffer_usm<u8>(
                    get_compute_scheduler_ptr(),
                    rnd,
                    shamalgs::primitives::mock_value<i32>(eng, 1, length)));
        }
    }

    shamalgs::collective::SerializedDDataComm send_data;

    dat_ref.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u8> &buf) {
        if (rank_owner[sender] == wrank) {
            send_data.add_obj(sender, receiver, buf.copy());
        }
    });

    shamalgs::collective::SerializedDDataComm recv_data;
    shamalgs::collective::DDSCommCache cache;
    distributed_data_sparse_comm(
        get_compute_scheduler_ptr(),
        send_data,
        recv_data,
        [&](u64 id) {
            return rank_owner[id];
        },
        cache,
        {},
        max_comm_size);

    shamalgs::collective::SerializedDDataComm recv_data_ref;
    dat_ref.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u8> &buf) {
        if (rank_owner[receiver] == wrank) {
            recv_data_ref.add_obj(sender, receiver, buf.copy());
        }
    });

    REQUIRE_EQUAL_NAMED(
        "expected number of recv",
        recv_data.get_element_count(),
        recv_data_ref.get_element_count());

    recv_data_ref.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u8> &buf) {
        REQUIRE_NAMED("has expected key", recv_data.has_key(sender, receiver));

        auto it = recv_data.get_native().find({sender, receiver});

        REQUIRE_NAMED(
            "correct buffer",
            shamalgs::primitives::equals(get_compute_scheduler_ptr(), buf, it->second));
    });
}

NEW_TEST(Unittest, "shamalgs/collective/distributedDataComm", -1) {

    const i32 wsize = shamcomm::world_size();
    const i32 wrank = shamcomm::world_rank();

    u32 npatch       = wsize * 5;
    u32 nbuf_p_patch = 2;
    u64 max_msg_len  = 1e4;

    u64 seed = 0x123;

    distribdata_sparse_comm_test(npatch, nbuf_p_patch, max_msg_len, seed, "", i32_max - 1);

    // test with lowered limit to test splitting
    distribdata_sparse_comm_test(
        npatch, nbuf_p_patch, max_msg_len, seed, "", 1e4); // max message len in test
}

NEW_TEST(Unittest, "shamalgs/collective/fetch_all_simple", -1) {

    using namespace shamalgs::collective;

    const i32 wsize = shamcomm::world_size();
    const i32 wrank = shamcomm::world_rank();

    u64 nid = 1000 * wsize;

    // ids handed to the ranks in a shuffled order so that global = allgatherv(local) is not
    // sorted by id
    std::vector<u64> ids(nid);
    for (u64 i = 0; i < nid; i++) {
        ids[i] = 3 * i + 7;
    }
    std::mt19937 eng(0x111);
    std::shuffle(ids.begin(), ids.end(), eng);

    auto value_of = [](u64 id) {
        return f64(id) * 0.5 + 1;
    };

    std::vector<u64> local_ids;
    shambase::DistributedData<f64> src;
    for (u64 i = 0; i < nid; i++) {
        if (i64(i % wsize) == wrank) {
            local_ids.push_back(ids[i]);
            src.add_obj(ids[i], value_of(ids[i]));
        }
    }

    std::vector<u64> global_ids;
    vector_allgatherv(
        local_ids, get_mpi_type<u64>(), global_ids, get_mpi_type<u64>(), MPI_COMM_WORLD);

    shambase::DistributedData<f64> ret
        = fetch_all_simple<f64, u64>(src, local_ids, global_ids, [](u64 id) {
              return id;
          });

    REQUIRE_EQUAL(ret.get_element_count(), nid);

    bool all_correct = true;
    for (u64 id : ids) {
        all_correct = all_correct && ret.has_key(id) && (ret.get(id) == value_of(id));
    }
    REQUIRE_NAMED("every id mapped to its own value", all_correct);
}

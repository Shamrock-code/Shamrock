// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamrock/legacy/comm/sparse_communicator.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include "shamtest/shamtest.hpp"

SchedulerPatchList make_plist(std::mt19937 &eng, u32 max_patch_per_node, u32 nb_part) {

    using namespace shamrock::patch;

    std::uniform_int_distribution<u32> dist_nb_patch(0, max_patch_per_node);
    u32 nb_patch_test = dist_nb_patch(eng);

    SchedulerPatchList plist;

    for (u32 i = 0; i < nb_patch_test; i++) {

        Patch p;
        p.id_patch      = i + u64(shamcomm::world_rank()) * max_patch_per_node;
        p.load_value    = nb_part;
        p.node_owner_id = shamcomm::world_rank();

        plist.local.push_back(p);
    }

    plist.build_global();
    plist.build_global_idx_map();

    return std::move(plist);
}

struct BenchResult {
    f64 nb_patch;
    f64 nb_obj;
    f64 fetch_time;
    f64 comm_time;
    f64 fetch_bandwidth;
    f64 comm_bandwidth;
};

BenchResult benchmark_comm(std::mt19937 &eng, u32 max_patch_per_node, u32 nb_part) {

    using namespace shamrock::patch;

    SchedulerPatchList plist = make_plist(eng, max_patch_per_node, nb_part);

    std::uniform_int_distribution<u32> dist_pb(0, plist.global.size() - 1);

    u64 total_comm_sz = 0;
    using T           = f32;
    std::vector<u64_2> send_vec;
    std::vector<std::unique_ptr<PatchDataField<T>>> send_data;

    for (const Patch &p : plist.local) {
        send_vec.push_back({plist.id_patch_to_global_idx[p.id_patch], dist_pb(eng)});

        std::unique_ptr<PatchDataField<T>> data = std::make_unique<PatchDataField<T>>("tmp", 1);
        data->gen_mock_data(p.load_value, eng);

        send_data.push_back(std::move(data));
    }

    SparsePatchCommunicator comm(plist.global, send_vec);

    mpi::barrier(MPI_COMM_WORLD);

    shambase::Timer fetch_timer;
    fetch_timer.start();
    comm.fetch_comm_table();
    fetch_timer.end();

    u64 xchg_fetch_data = comm.get_xchg_byte_count();
    comm.reset_xchg_byte_count();

    mpi::barrier(MPI_COMM_WORLD);

    shambase::Timer comm_timer;
    comm_timer.start();
    SparseCommResult<PatchDataField<T>> out = comm.sparse_exchange(send_data);
    mpi::barrier(MPI_COMM_WORLD);
    comm_timer.end();

    u64 xchg_comm_data = comm.get_xchg_byte_count();
    comm.reset_xchg_byte_count();

    u64 nb_obj = 0;
    for (const Patch &p : plist.global) {
        nb_obj += p.load_value;
    }

    u64 sum_byte_fetch, sum_byte_comm;

    mpi::reduce(
        &xchg_fetch_data, &sum_byte_fetch, 1, get_mpi_type<u64>(), MPI_SUM, 0, MPI_COMM_WORLD);
    mpi::reduce(
        &xchg_comm_data, &sum_byte_comm, 1, get_mpi_type<u64>(), MPI_SUM, 0, MPI_COMM_WORLD);

    BenchResult bench;

    bench.nb_patch        = plist.global.size();
    bench.nb_obj          = nb_obj;
    bench.fetch_time      = fetch_timer.nanosec / 1e9;
    bench.comm_time       = comm_timer.nanosec / 1e9;
    bench.fetch_bandwidth = sum_byte_fetch / bench.fetch_time;
    bench.comm_bandwidth  = sum_byte_comm / bench.comm_time;

    return bench;
}

TestStart(Benchmark, "core/comm/sparse_communicator_patchdata_field:", func_name, -1) {

    u32 nb_part_per_patch = 1e5;

    std::mt19937 eng(0x2525 + shamcomm::world_rank());

    auto &bandwidth_dataset = shamtest::test_data().new_dataset("bandwidth");

    std::vector<f64> nb_patch;
    std::vector<f64> nb_obj;
    std::vector<f64> fetch_time;
    std::vector<f64> comm_time;
    std::vector<f64> fetch_bandwidth;
    std::vector<f64> comm_bandwidth;

    u64 max_mem_sz = 2e9;

    f64 max_patch_per_node = 1;

    u32 n_max_sample = 40;
    for (u32 i = 0; i < n_max_sample; i++) {

        max_patch_per_node *= 1.2;

        if (u64(max_patch_per_node) * u64(nb_part_per_patch) > max_mem_sz) {
            break;
        }

        shamlog_debug_ln(
            "Test",
            i,
            "/",
            n_max_sample,
            " | testing comm : ",
            u32(max_patch_per_node),
            nb_part_per_patch);

        auto res = benchmark_comm(eng, u32(max_patch_per_node), nb_part_per_patch);
        nb_patch.push_back(res.nb_patch);
        nb_obj.push_back(res.nb_obj);
        fetch_time.push_back(res.fetch_time);
        comm_time.push_back(res.comm_time);
        fetch_bandwidth.push_back(res.fetch_bandwidth);
        comm_bandwidth.push_back(res.comm_bandwidth);
    }

    if (shamcomm::world_rank() == 0) {
        bandwidth_dataset.add_data("nb_patch", nb_patch);
        bandwidth_dataset.add_data("nb_obj", nb_obj);
        bandwidth_dataset.add_data("fetch_time", fetch_time);
        bandwidth_dataset.add_data("comm_time", comm_time);
        bandwidth_dataset.add_data("fetch_bandwidth", fetch_bandwidth);
        bandwidth_dataset.add_data("comm_bandwidth", comm_bandwidth);
    }
}

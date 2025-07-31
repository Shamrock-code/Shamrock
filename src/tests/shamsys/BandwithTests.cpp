// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"

template<class T>
void bench_memcpy_sycl(
    std::string dset_name, sycl::queue &q1, sycl::queue &q2, u64 max_byte_sz_cnt) {

    shamlog_debug_ln("bench_memcpy_sycl", dset_name);

    std::vector<f64> sz;
    std::vector<f64> bandwidth_GBsm1;

    using namespace shamsys::instance;

    u64 max_obj_cnt = max_byte_sz_cnt / sizeof(T);

    for (f64 cnt_ = 1e2; cnt_ < max_obj_cnt; cnt_ *= 1.1) {

        u64 cnt = cnt_;

        auto ptr1 = sycl::malloc_device<T>(cnt, q1);
        auto ptr2 = sycl::malloc_device<T>(cnt, q2);

        get_compute_queue().wait();

        shambase::Timer t;
        t.start();
        get_compute_queue().memcpy(ptr1, ptr2, cnt * sizeof(T)).wait();
        t.end();

        sycl::free(ptr1, q1);
        sycl::free(ptr2, q2);

        sz.push_back(cnt * sizeof(T) / f64(1024 * 1024 * 1024));
        bandwidth_GBsm1.push_back(
            (f64(cnt * sizeof(T)) / (t.nanosec / 1e9)) / f64(1024 * 1024 * 1024));
    }

    auto &dset = shamtest::test_data().new_dataset(dset_name);

    dset.add_data("size (GB)", sz);
    dset.add_data("bandwidth (GB.s^-1)", bandwidth_GBsm1);
}

template<class T>
void bench_memcpy_sycl_host_dev(std::string dset_name, sycl::queue &q1, u64 max_byte_sz_cnt) {

    shamlog_debug_ln("bench_memcpy_sycl_host_dev", dset_name);

    std::vector<f64> sz;
    std::vector<f64> bandwidth_GBsm1;

    using namespace shamsys::instance;

    u64 max_obj_cnt = max_byte_sz_cnt / sizeof(T);

    for (f64 cnt_ = 1e2; cnt_ < max_obj_cnt; cnt_ *= 1.1) {

        u64 cnt = cnt_;

        auto ptr1 = sycl::malloc_device<T>(cnt, q1);
        auto ptr2 = sycl::malloc_host<T>(cnt, q1);

        get_compute_queue().wait();

        shambase::Timer t;
        t.start();
        get_compute_queue().memcpy(ptr1, ptr2, cnt * sizeof(T)).wait();
        t.end();

        sycl::free(ptr1, q1);
        sycl::free(ptr2, q1);

        sz.push_back(cnt * sizeof(T) / f64(1024 * 1024 * 1024));
        bandwidth_GBsm1.push_back(
            (f64(cnt * sizeof(T)) / (t.nanosec / 1e9)) / f64(1024 * 1024 * 1024));
    }

    auto &dset = shamtest::test_data().new_dataset(dset_name);

    dset.add_data("size (GB)", sz);
    dset.add_data("bandwidth (GB.s^-1)", bandwidth_GBsm1);
}

TestStart(Benchmark, "bandwidth-tests/memcpy-sycl", compcompbandwidthtest, 1) {
    using namespace shamsys::instance;

    u64 max_sz = get_compute_device().get_info<sycl::info::device::global_mem_size>() / 16;

    bench_memcpy_sycl<f32>("comp -> comp (f32)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f64>("comp -> comp (f64)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<u32>("comp -> comp (u32)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<u64>("comp -> comp (u64)", get_compute_queue(), get_compute_queue(), max_sz);

    bench_memcpy_sycl<f32_2>(
        "comp -> comp (f32_2)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f32_3>(
        "comp -> comp (f32_3)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f32_4>(
        "comp -> comp (f32_4)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f32_8>(
        "comp -> comp (f32_8)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f32_16>(
        "comp -> comp (f32_16)", get_compute_queue(), get_compute_queue(), max_sz);

    bench_memcpy_sycl<f64_2>(
        "comp -> comp (f64_2)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f64_3>(
        "comp -> comp (f64_3)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f64_4>(
        "comp -> comp (f64_4)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f64_8>(
        "comp -> comp (f64_8)", get_compute_queue(), get_compute_queue(), max_sz);
    bench_memcpy_sycl<f64_16>(
        "comp -> comp (f64_16)", get_compute_queue(), get_compute_queue(), max_sz);

    max_sz /= 16;

    bench_memcpy_sycl_host_dev<f32>("comp -> host (f32)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f64>("comp -> host (f64)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<u32>("comp -> host (u32)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<u64>("comp -> host (u64)", get_compute_queue(), max_sz);

    bench_memcpy_sycl_host_dev<f32_2>("comp -> host (f32_2)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f32_3>("comp -> host (f32_3)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f32_4>("comp -> host (f32_4)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f32_8>("comp -> host (f32_8)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f32_16>("comp -> host (f32_16)", get_compute_queue(), max_sz);

    bench_memcpy_sycl_host_dev<f64_2>("comp -> host (f64_2)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f64_3>("comp -> host (f64_3)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f64_4>("comp -> host (f64_4)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f64_8>("comp -> host (f64_8)", get_compute_queue(), max_sz);
    bench_memcpy_sycl_host_dev<f64_16>("comp -> host (f64_16)", get_compute_queue(), max_sz);
}

// MPI comm

template<class T>
void make_bandwidth_matrix(std::string dset_name, sycl::queue &q1, u32 comm_size) {

    using namespace shamsys::instance;

    T *ptr1;
    T *ptr2;

    auto comm = [&](u32 rank_send, u32 rank_recv) -> f64 {
        mpi::barrier(MPI_COMM_WORLD);

        shambase::Timer t;
        t.start();

        MPI_Request rq_send, rq_recv;

        if (shamcomm::world_rank() == rank_send) {
            shamcomm::mpi::Isend(
                ptr1, comm_size, get_mpi_type<T>(), rank_recv, 0, MPI_COMM_WORLD, &rq_send);
        }

        if (shamcomm::world_rank() == rank_recv) {
            shamcomm::mpi::Irecv(
                ptr2, comm_size, get_mpi_type<T>(), rank_send, 0, MPI_COMM_WORLD, &rq_recv);
        }

        if (shamcomm::world_rank() == rank_send) {
            MPI_Status st;
            shamcomm::mpi::Wait(&rq_send, &st);
        }

        if (shamcomm::world_rank() == rank_recv) {
            MPI_Status st;
            shamcomm::mpi::Wait(&rq_recv, &st);
        }

        mpi::barrier(MPI_COMM_WORLD);

        t.end();

        return (f64(comm_size * sizeof(T)) / (t.nanosec / 1e9)) / f64(1024 * 1024 * 1024);
    };

    auto test = [&](std::string dset_name) {
        std::vector<f64> rank_send;
        std::vector<f64> rank_recv;
        std::vector<f64> bw;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            for (u32 j = 0; j < shamcomm::world_size(); j++) {

                shamlog_debug_ln("make_bandwidth_matrix", i, "->", j, dset_name);

                rank_send.push_back(i);
                rank_recv.push_back(j);

                f64 tot  = 0;
                u32 nrep = 4;
                for (u32 rep = 0; rep < nrep; rep++) {
                    tot += comm(i, j);
                }

                bw.push_back(tot / nrep);
            }
        }

        auto &dset = shamtest::test_data().new_dataset(dset_name);
        dset.add_data("rank_send", rank_send);
        dset.add_data("rank_recv", rank_recv);
        dset.add_data("bandwidth (GB.s^-1)", bw);
    };

    ptr1 = sycl::malloc_device<T>(comm_size, q1);
    ptr2 = sycl::malloc_device<T>(comm_size, q1);
    q1.wait();
    test(dset_name + " device->device (" + shambase::readable_sizeof(comm_size * sizeof(T)) + ")");
    sycl::free(ptr1, q1);
    sycl::free(ptr2, q1);

    ptr1 = sycl::malloc_host<T>(comm_size, q1);
    ptr2 = sycl::malloc_device<T>(comm_size, q1);
    q1.wait();
    test(dset_name + " host->device (" + shambase::readable_sizeof(comm_size * sizeof(T)) + ")");
    sycl::free(ptr1, q1);
    sycl::free(ptr2, q1);

    ptr1 = sycl::malloc_device<T>(comm_size, q1);
    ptr2 = sycl::malloc_host<T>(comm_size, q1);
    q1.wait();
    test(dset_name + " device->host (" + shambase::readable_sizeof(comm_size * sizeof(T)) + ")");
    sycl::free(ptr1, q1);
    sycl::free(ptr2, q1);

    ptr1 = sycl::malloc_host<T>(comm_size, q1);
    ptr2 = sycl::malloc_host<T>(comm_size, q1);
    q1.wait();
    test(dset_name + " host->host (" + shambase::readable_sizeof(comm_size * sizeof(T)) + ")");
    sycl::free(ptr1, q1);
    sycl::free(ptr2, q1);
}

TestStart(Benchmark, "bandwidth-tests/mpi-pair-comm/bw-matrix", mpi_pair_comm_bw_matrix, -1) {
    make_bandwidth_matrix<f32>("(f32)", shamsys::instance::get_compute_queue(), 1024 * 1024 * 16);
    make_bandwidth_matrix<f32_3>(
        "(f32_3)", shamsys::instance::get_compute_queue(), 1024 * 1024 * 16 / 256);
}

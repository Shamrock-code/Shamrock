// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MicroBenchmark.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/benchmarks/add_mul.hpp"
#include "shambackends/benchmarks/saxpy.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <mpi.h>
#include <stdexcept>

namespace shamsys::microbench {
    void p2p_bandwith(u32 wr_sender, u32 wr_receiv);
    void p2p_latency(u32 wr1, u32 wr2);
    void saxpy();
    void add_mul_rotation();
} // namespace shamsys::microbench

void shamsys::run_micro_benchmark() {
    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln("Running micro benchamrks :");
    }

    u32 wr1 = 0;
    u32 wr2 = shamcomm::world_size() - 1;

    microbench::p2p_bandwith(wr1, wr2);
    if (shamcomm::world_size() > 1) {
        microbench::p2p_latency(wr1, wr2);
    }
    microbench::saxpy();
    microbench::add_mul_rotation();
}

void shamsys::microbench::p2p_bandwith(u32 wr_sender, u32 wr_receiv) {
    StackEntry stack_loc{};

    u32 wr = shamcomm::world_rank();

    u64 length = 1024UL * 1014UL * 8UL; // 8MB messages
    shamcomm::CommunicationBuffer buf_recv{length, instance::get_compute_scheduler_ptr()};
    shamcomm::CommunicationBuffer buf_send{length, instance::get_compute_scheduler_ptr()};

    std::vector<MPI_Request> rqs;

    f64 t        = 0;
    u64 loops    = 0;
    bool is_used = false;
    do {
        loops++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if (wr == wr_sender) {
            rqs.push_back(MPI_Request{});
            u32 rq_index = rqs.size() - 1;
            auto &rq     = rqs[rq_index];
            mpi::isend(buf_send.get_ptr(), length, MPI_BYTE, wr_receiv, 0, MPI_COMM_WORLD, &rq);
            is_used = true;
        }

        if (wr == wr_receiv) {
            MPI_Status s;
            mpi::recv(buf_recv.get_ptr(), length, MPI_BYTE, wr_sender, 0, MPI_COMM_WORLD, &s);
            is_used = true;
        }

        if (!is_used) {
            t = 1;
        }
        std::vector<MPI_Status> st_lst(rqs.size());
        if (rqs.size() > 0) {
            mpi::waitall(rqs.size(), rqs.data(), st_lst.data());
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

    } while (shamalgs::collective::allreduce_min(t) < 1);

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(shambase::format(
            " - p2p bandwith : {:0.2f} GB.s^-1 (ranks : {} -> {}) (loops : {})",
            (f64(length * loops) / t) * 1e-9,
            wr_sender,
            wr_receiv,
            loops));
    }
}

void shamsys::microbench::p2p_latency(u32 wr1, u32 wr2) {
    StackEntry stack_loc{};

    if (wr1 == wr2) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "can not launch this test with same ranks");
    }

    u32 wr = shamcomm::world_rank();

    u64 length = 8ULL; // 8B messages
    shamcomm::CommunicationBuffer buf_recv{length, instance::get_compute_scheduler_ptr()};
    shamcomm::CommunicationBuffer buf_send{length, instance::get_compute_scheduler_ptr()};

    f64 t        = 0;
    u64 loops    = 0;
    bool is_used = false;
    do {
        loops++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if (wr == wr1) {
            MPI_Status s;
            mpi::send(buf_send.get_ptr(), length, MPI_BYTE, wr2, 0, MPI_COMM_WORLD);
            mpi::recv(buf_recv.get_ptr(), length, MPI_BYTE, wr2, 1, MPI_COMM_WORLD, &s);
            is_used = true;
        }

        if (wr == wr2) {
            MPI_Status s;
            mpi::recv(buf_recv.get_ptr(), length, MPI_BYTE, wr1, 0, MPI_COMM_WORLD, &s);
            mpi::send(buf_send.get_ptr(), length, MPI_BYTE, wr1, 1, MPI_COMM_WORLD);
            is_used = true;
        }

        if (!is_used) {
            t = 1;
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

    } while (shamalgs::collective::allreduce_min(t) < 1);

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(shambase::format(
            " - p2p latency  : {:.4e} s (ranks : {} <-> {}) (loops : {})",
            t / f64(loops),
            wr1,
            wr2,
            loops));
    }
}

void shamsys::microbench::saxpy() {
    int N = (1 << 27);

    using vec4    = sycl::vec<float, 4>;
    int vec4_size = sizeof(vec4);
    auto result   = sham::benchmarks::saxpy_bench<vec4>(
        instance::get_compute_scheduler_ptr(),
        N,
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        vec4_size);

    f64 min_bw = shamalgs::collective::allreduce_min(result.bandwidth);
    f64 max_bw = shamalgs::collective::allreduce_max(result.bandwidth);
    f64 sum_bw = shamalgs::collective::allreduce_sum(result.bandwidth);
    f64 avg_bw = sum_bw / (f64) shamcomm::world_size();

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(shambase::format(
            " - saxpy : {:.2f} GB.s^-1 (min = {:.2f}, max = {:.2f}, avg = {:.2f})",
            sum_bw,
            min_bw,
            max_bw,
            avg_bw));
    }
}

void shamsys::microbench::add_mul_rotation() {
    int N = (1 << 20);

    using vec4 = sycl::vec<float, 4>;

    auto result = sham::benchmarks::add_mul_bench<vec4>(
        instance::get_compute_scheduler_ptr(),
        N,
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        {cos(2.0f), cos(2.0f), cos(2.0f), cos(2.0f)},
        {sin(2.0f), sin(2.0f), sin(2.0f), sin(2.0f)},
        10000,
        4);

    f64 gflops = result.flops / 1e9;

    f64 min_flop = shamalgs::collective::allreduce_min(gflops);
    f64 max_flop = shamalgs::collective::allreduce_max(gflops);
    f64 sum_flop = shamalgs::collective::allreduce_sum(gflops);
    f64 avg_flop = sum_flop / (f64) shamcomm::world_size();

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(shambase::format(
            " - add_mul : {:.2f} Gflops (min = {:.2f}, max = {:.2f}, avg = {:.2f})",
            sum_flop,
            min_flop,
            max_flop,
            avg_flop));
    }
}

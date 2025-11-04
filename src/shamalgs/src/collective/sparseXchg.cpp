// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparseXchg.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/sparseXchg.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

    auto get_hash_comm_map = [](const std::vector<u64> &vec) {
        std::string s = "";
        s.resize(vec.size() * sizeof(u64));
        std::memcpy(s.data(), vec.data(), vec.size() * sizeof(u64));
        auto ret = std::hash<std::string>{}(s);
        return ret;
    };

    auto check_comm_hash = [](const std::vector<u64> &vec) {
        auto hash = get_hash_comm_map(vec);
        // logger::raw_ln("global_comm_ranks hash", hash);

        auto max_hash = shamalgs::collective::allreduce_max(hash);
        auto min_hash = shamalgs::collective::allreduce_min(hash);

        if (max_hash != min_hash) {
            std::string msg = shambase::format(
                "hash mismatch {} != {}, local hash = {}", max_hash, min_hash, hash);
            logger::err_ln("Sparse comm", msg);
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
            shambase::throw_with_loc<std::runtime_error>(msg);
        }
    };

    // Utility lambda for error reporting
    auto check_payload_size_is_int = [](u64 bytesz, const std::vector<u64> &global_comm_ranks) {
        u64 payload_sz = bytesz;

        if (payload_sz > std::numeric_limits<i32>::max()) {

            std::vector<u64> send_sizes;
            for (u32 i = 0; i < global_comm_ranks.size(); i++) {
                u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

                if (comm_ranks.x() == shamcomm::world_rank()) {
                    send_sizes.push_back(payload_sz);
                }
            }

            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "payload size {} is too large for MPI (max i32 is {})\n"
                "message sizes to send: {}",
                payload_sz,
                std::numeric_limits<i32>::max(),
                send_sizes));
        }

        return (i32) payload_sz;
    };

    struct rq_info {
        i32 sender;
        i32 receiver;
        u64 size;
        i32 tag;
        bool is_send;
        bool is_recv;
    };

    struct RequestList {

        std::vector<MPI_Request> rqs;
        std::vector<bool> is_ready;

        u32 ready_count = 0;

        MPI_Request &new_request() {
            rqs.push_back(MPI_Request{});
            u32 rq_index = rqs.size() - 1;
            auto &rq     = rqs[rq_index];
            is_ready.push_back(false);
            return rq;
        }

        void test_ready() {
            for (u32 i = 0; i < rqs.size(); i++) {
                if (!is_ready[i]) {
                    MPI_Status st;
                    int ready;
                    shamcomm::mpi::Test(&rqs[i], &ready, MPI_STATUS_IGNORE);
                    if (ready) {
                        is_ready[i] = true;
                        ready_count++;
                    }
                }
            }
        }

        bool all_ready() { return ready_count == rqs.size(); }

        void wait_all() {
            std::vector<MPI_Status> st_lst(rqs.size());
            shamcomm::mpi::Waitall(rqs.size(), rqs.data(), st_lst.data());
        }

        u32 remain_count() {
            test_ready();
            return rqs.size() - ready_count;
        }
    };

    auto report_unfinished_requests = [](RequestList &rqs, std::vector<rq_info> &rqs_infos) {
        std::string err_msg = "";
        for (u32 i = 0; i < rqs.rqs.size(); i++) {
            if (!rqs.is_ready[i]) {

                if (rqs_infos[i].is_send) {
                    err_msg += shambase::format(
                        "communication timeout : send {} -> {} tag {} size {}\n",
                        rqs_infos[i].sender,
                        rqs_infos[i].receiver,
                        rqs_infos[i].tag,
                        rqs_infos[i].size);
                } else {
                    err_msg += shambase::format(
                        "communication timeout : recv {} -> {} tag {} size {}\n",
                        rqs_infos[i].sender,
                        rqs_infos[i].receiver,
                        rqs_infos[i].tag,
                        rqs_infos[i].size);
                }
            }
        }
        std::string msg = shambase::format("communication timeout : \n{}", err_msg);
        logger::err_ln("Sparse comm", msg);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        shambase::throw_with_loc<std::runtime_error>(msg);
    };

    auto test_event_completions
        = [](std::vector<MPI_Request> &rqs, std::vector<rq_info> &rqs_infos) {
              shambase::Timer twait;
              twait.start();
              f64 timeout_t  = 120;
              f64 freq_print = 10;

              std::vector<bool> done_map = {};
              done_map.resize(rqs.size());
              for (u32 i = 0; i < rqs.size(); i++) {
                  done_map[i] = false;
              }

              f64 t_last_print = 0;
              u64 done_count   = 0;

              bool done = false;
              while (!done) {
                  bool loc_done = true;
                  for (u32 i = 0; i < rqs.size(); i++) {
                      if (done_map[i]) {
                          continue;
                      }

                      auto &rq = rqs[i];

                      MPI_Status st;
                      int ready;
                      shamcomm::mpi::Test(&rq, &ready, MPI_STATUS_IGNORE);
                      if (!ready) {
                          loc_done = false;
                          // logger::raw_ln(shambase::format(
                          //     "communication pending : send {} -> {} tag {} size {}",
                          //     rqs_infos[i].sender,
                          //     rqs_infos[i].receiver,
                          //     rqs_infos[i].tag,
                          //     rqs_infos[i].size));
                      } else {
                          done_map[i] = true;
                          done_count++;
                          // logger::raw_ln(shambase::format(
                          //     "communication done : send {} -> {} tag {} size {}",
                          //     rqs_infos[i].sender,
                          //     rqs_infos[i].receiver,
                          //     rqs_infos[i].tag,
                          //     rqs_infos[i].size));
                      }
                  }

                  if (loc_done) {
                      done = true;
                  }

                  twait.end();

                  if (twait.elasped_sec() > t_last_print + 10) {

                      std::string msg
                          = shambase::format("Sparse comm : {} / {} done", done_count, rqs.size());
                      logger::warn_ln("Sparse comm", msg);

                      t_last_print = twait.elasped_sec();
                  }

                  if (twait.elasped_sec() > timeout_t) {
                      std::string err_msg = "";
                      for (u32 i = 0; i < rqs.size(); i++) {
                          if (!done_map[i]) {

                              if (rqs_infos[i].is_send) {
                                  err_msg += shambase::format(
                                      "communication timeout : send {} -> {} tag {} size {}\n",
                                      rqs_infos[i].sender,
                                      rqs_infos[i].receiver,
                                      rqs_infos[i].tag,
                                      rqs_infos[i].size);
                              } else {
                                  err_msg += shambase::format(
                                      "communication timeout : recv {} -> {} tag {} size {}\n",
                                      rqs_infos[i].sender,
                                      rqs_infos[i].receiver,
                                      rqs_infos[i].tag,
                                      rqs_infos[i].size);
                              }
                          }
                      }
                      std::string msg = shambase::format("communication timeout : \n{}", err_msg);
                      logger::err_ln("Sparse comm", msg);
                      std::this_thread::sleep_for(std::chrono::seconds(2));
                      shambase::throw_with_loc<std::runtime_error>(msg);
                  }
              }
          };
} // namespace

auto get_SHAM_SPARSE_COMM_INFLIGHT_LIM = []() {
    std::string val = shamcmdopt::getenv_str_default_register(
        "SHAM_SPARSE_COMM_INFLIGHT_LIM", "128", "Maximum number of inflight messages");

    u64 ret = 128;
    try {
        ret = std::stoull(val);
    } catch (...) {
        logger::err_ln(
            "Sparse comm",
            shambase::format(
                "Invalid value for SHAM_SPARSE_COMM_INFLIGHT_LIM {}, using default value {}",
                val,
                ret));
    }

    return ret;
};

const u64 SHAM_SPARSE_COMM_INFLIGHT_LIM = get_SHAM_SPARSE_COMM_INFLIGHT_LIM();

namespace shamalgs::collective {
    void sparse_comm_debug_infos(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // Utility lambda for printing comm matrix
        auto print_comm_mat = [&]() {
            StackEntry stack_loc{};

            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
            std::string accum = "";

            u32 send_idx = 0;
            for (u32 i = 0; i < global_comm_ranks.size(); i++) {
                u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

                if (comm_ranks.x() == shamcomm::world_rank()) {
                    accum += shambase::format(
                        "{} # {} # {}\n",
                        comm_ranks.x(),
                        comm_ranks.y(),
                        message_send[send_idx].payload->get_size());

                    send_idx++;
                }
            }

            std::string matrix;
            shamcomm::gather_str(accum, matrix);

            matrix = "\n" + matrix;

            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("comm matrix:", matrix);
            }
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        };

        // Enable this only to do debug
        print_comm_mat();

        auto show_alloc_state = [&]() {
            StackEntry stack_loc{};
            sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

            std::string accum = shambase::format(
                "rank = {} maxmem = {}\n",
                shamcomm::world_rank(),
                shambase::readable_sizeof(mem_perf_infos_end.max_allocated_byte_device));

            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
            std::string log;
            shamcomm::gather_str(accum, log);

            log = "\n" + log;

            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("alloc state:", log);
            }
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        };

        // Enable this only to do debug
        show_alloc_state();
    }

    void sparse_comm_isend_probe_count_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        // send step
        u32 send_idx = 0;
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.x() == shamcomm::world_rank()) {

                auto &payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                int send_sz = check_payload_size_is_int(payload->get_size(), global_comm_ranks);

                // logger::raw_ln(shambase::format(
                //     "[{}] send {} bytes to rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     payload->get_bytesize(),
                //     comm_ranks.y(),
                //     i));

                shamcomm::mpi::Isend(
                    payload->get_ptr(), send_sz, MPI_BYTE, comm_ranks.y(), i, MPI_COMM_WORLD, &rq);

                send_idx++;
            }
        }

        // recv step
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.y() == shamcomm::world_rank()) {

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                MPI_Status st;
                i32 cnt;
                shamcomm::mpi::Probe(comm_ranks.x(), i, MPI_COMM_WORLD, &st);
                shamcomm::mpi::Get_count(&st, MPI_BYTE, &cnt);

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, dev_sched);

                // logger::raw_ln(shambase::format(
                //     "[{}] recv {} bytes from rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     cnt,
                //     comm_ranks.x(),
                //     i));

                shamcomm::mpi::Irecv(
                    payload.payload->get_ptr(),
                    cnt,
                    MPI_BYTE,
                    comm_ranks.x(),
                    i,
                    MPI_COMM_WORLD,
                    &rq);

                message_recv.push_back(std::move(payload));
            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        shamcomm::mpi::Waitall(rqs.size(), rqs.data(), st_lst.data());
    }

    void sparse_comm_allgather_isend_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // check hash
        // check_comm_hash(global_comm_ranks);

        // Build global comm size table
        std::vector<int> comm_sizes_loc = {};
        comm_sizes_loc.resize(message_send.size());
        for (u64 i = 0; i < message_send.size(); i++) {
            comm_sizes_loc[i]
                = check_payload_size_is_int(message_send[i].payload->get_size(), global_comm_ranks);
        }

        // gather sizes
        std::vector<int> comm_sizes = {};
        vector_allgatherv(comm_sizes_loc, comm_sizes, MPI_COMM_WORLD);

        // Init the receiving buffers
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            i32 sender   = comm_ranks.x();
            i32 receiver = comm_ranks.y();

            if (receiver == shamcomm::world_rank()) {
                RecvPayload payload;
                payload.sender_ranks = sender;
                i32 cnt              = comm_sizes[i];

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, dev_sched);

                message_recv.push_back(std::move(payload));
            }
        }

        RequestList rqs;
        std::vector<rq_info> rqs_infos;

        std::vector<i32> tag_map(shamcomm::world_size(), 0);

        // send step
        u32 send_idx = 0;
        u32 recv_idx = 0;

        u32 in_flight = 0;

        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            i32 sender   = comm_ranks.x();
            i32 receiver = comm_ranks.y();

            i32 tag = tag_map[sender];
            tag_map[sender]++;

            bool trigger_check = false;

            if (sender == shamcomm::world_rank()) {

                auto &payload = message_send.at(send_idx).payload;

                auto &rq = rqs.new_request();

                rqs_infos.push_back({sender, receiver, payload->get_size(), tag, true, false});

                SHAM_ASSERT(payload->get_size() == comm_sizes_loc[send_idx]);

                // logger::raw_ln(shambase::format(
                //     "[{}] send {} bytes to rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     payload->get_bytesize(),
                //     comm_ranks.y(),
                //     i));

                shamcomm::mpi::Isend(
                    payload->get_ptr(),
                    comm_sizes_loc[send_idx],
                    MPI_BYTE,
                    receiver,
                    tag,
                    MPI_COMM_WORLD,
                    &rq);

                send_idx++;
                in_flight++;
            }

            if (receiver == shamcomm::world_rank()) {

                auto &payload = message_recv.at(recv_idx).payload;

                auto &rq = rqs.new_request();

                rqs_infos.push_back({sender, receiver, u64(comm_sizes[i]), tag, false, true});

                // logger::raw_ln(shambase::format(
                //     "[{}] recv {} bytes from rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     cnt,
                //     comm_ranks.x(),
                //     i));

                shamcomm::mpi::Irecv(
                    payload->get_ptr(), comm_sizes[i], MPI_BYTE, sender, tag, MPI_COMM_WORLD, &rq);

                recv_idx++;
                in_flight++;
            }

            // routine to limit the number of in-flight messages
            u64 in_flight_lim = SHAM_SPARSE_COMM_INFLIGHT_LIM;
            if (in_flight > in_flight_lim) {

                f64 timeout    = 120; // seconds
                f64 print_freq = 10;  // seconds

                f64 last_print_time = 0;

                shambase::Timer twait;
                twait.start();
                do {
                    twait.end();
                    if (twait.elasped_sec() > timeout) {
                        report_unfinished_requests(rqs, rqs_infos);
                    }

                    if (twait.elasped_sec() - last_print_time > print_freq) {
                        logger::warn_ln(
                            "SparseComm",
                            "too many messages in flight :",
                            in_flight,
                            "/",
                            in_flight_lim);
                        last_print_time = twait.elasped_sec();
                    }
                    in_flight = rqs.remain_count();
                } while (in_flight > in_flight_lim);
            }
        }

        test_event_completions(rqs.rqs, rqs_infos);

        rqs.wait_all();

        // logger::raw_ln(tag_map);

        // shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        // if (shamcomm::world_rank() == 0) {
        //     logger::raw_ln(shambase::format("sparse comm done"));
        // }
        // shamcomm::mpi::Barrier(MPI_COMM_WORLD);
    }
} // namespace shamalgs::collective

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RequestList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Provides a helper class to manage a list of MPI requests.
 *
 */

#include "shambase/narrowing.hpp"
#include "shambase/time.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/wrapper.hpp"
#include <vector>

namespace shamalgs::collective {

    class RequestList {

        std::vector<MPI_Request> rqs;
        std::vector<bool> is_ready;

        size_t ready_count = 0;

        public:
        MPI_Request &new_request() {
            rqs.emplace_back();
            is_ready.push_back(false);
            return rqs.back();
        }

        size_t size() const { return rqs.size(); }
        bool is_event_ready(size_t i) const { return is_ready[i]; }
        std::vector<MPI_Request> &requests() { return rqs; }

        void test_ready() {
            for (size_t i = 0; i < rqs.size(); i++) {
                if (!is_ready[i]) {
                    int ready;
                    shamcomm::mpi::Test(&rqs[i], &ready, MPI_STATUS_IGNORE);
                    if (ready) {
                        is_ready[i] = true;
                        ready_count++;
                    }
                }
            }
        }

        bool all_ready() const { return ready_count == rqs.size(); }

        void wait_all() {
            if (ready_count == rqs.size()) {
                return;
            }
            std::vector<MPI_Status> st_lst(rqs.size());
            shamcomm::mpi::Waitall(
                shambase::narrow_or_throw<i32>(rqs.size()), rqs.data(), st_lst.data());
            ready_count = rqs.size();
            is_ready.assign(rqs.size(), true);
        }

        size_t remain_count_no_test() { return rqs.size() - ready_count; }

        size_t remain_count() {
            test_ready();
            return rqs.size() - ready_count;
        }

        void report_timeout() const {
            std::string err_msg = "";
            for (size_t i = 0; i < rqs.size(); i++) {
                if (!is_ready[i]) {
                    err_msg += shambase::format("request {} is not ready\n", i);
                }
            }
            std::string msg = shambase::format("timeout : \n{}", err_msg);
            throw shambase::make_except_with_loc<std::runtime_error>(msg);
        }

        // spin lock until the number of in-flight requests is less than max_in_flight
        void spin_lock_partial_wait(size_t max_in_flight, f64 timeout, f64 print_freq) {

            if (rqs.size() < max_in_flight) {
                return;
            }

            shambase::Timer twait;
            twait.start();
            f64 last_print_time = 0;
            size_t in_flight;

            while ((in_flight = remain_count()) >= max_in_flight) {
                twait.end();
                if (twait.elasped_sec() > timeout) {
                    report_timeout();
                }

                if (twait.elasped_sec() - last_print_time > print_freq) {
                    logger::warn_ln(
                        "SparseComm",
                        "too many messages in flight :",
                        in_flight,
                        "/",
                        max_in_flight);
                    last_print_time = twait.elasped_sec();
                }
            }
        }
    };

} // namespace shamalgs::collective

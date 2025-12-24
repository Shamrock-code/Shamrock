// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RequestList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambase/time.hpp"
#include "shamcomm/wrapper.hpp"
#include <vector>

namespace shamalgs::collective {

    class RequestList {

        std::vector<MPI_Request> rqs;
        std::vector<bool> is_ready;

        size_t ready_count = 0;

        public:
        MPI_Request &new_request() {
            rqs.push_back(MPI_Request{});
            size_t rq_index = rqs.size() - 1;
            auto &rq        = rqs[rq_index];
            is_ready.push_back(false);
            return rq;
        }

        size_t size() { return rqs.size(); }
        bool is_event_ready(size_t i) { return is_ready[i]; }
        std::vector<MPI_Request> &requests() { return rqs; }

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
            shamcomm::mpi::Waitall(
                shambase::narrow_or_throw<i32>(rqs.size()), rqs.data(), st_lst.data());
        }

        size_t remain_count() {
            test_ready();
            return rqs.size() - ready_count;
        }
    };

} // namespace shamalgs::collective

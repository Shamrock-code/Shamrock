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
 * @brief Provides a helper class to manage a list of MPI requests.
 *
 */

#include "shambase/narrowing.hpp"
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

        size_t remain_count() {
            test_ready();
            return rqs.size() - ready_count;
        }
    };

} // namespace shamalgs::collective

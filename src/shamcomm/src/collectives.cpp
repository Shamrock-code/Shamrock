// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file collectives.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <unordered_map>
#include <vector>

namespace {

    /**
     * @brief Gather a vector of characters from all MPI ranks into a single string at root rank
     *
     * @param send_vec The string to send from each rank
     * @param recv_vec The resulting string at root rank
     *
     * @details This function is only available if `MPI_COMM_WORLD` is defined.
     * If `MPI_COMM_WORLD` is not defined, the function will not be available.
     *
     * @warning This function is not thread-safe.
     */
    template<class Tchar>
    inline void _internal_gather_str(
        const std::basic_string<Tchar> &send_vec, std::basic_string<Tchar> &recv_vec) {
        StackEntry stack_loc{};

        if (shamcomm::world_size() == 1) {
            recv_vec = send_vec;
            return;
        }

        u32 wsize = shamcomm::world_size();

        std::vector<int> counts(wsize);
        std::vector<int> disps(wsize);

        u32 local_count = send_vec.size();

        shamcomm::mpi::Gather(&local_count, 1, MPI_INT, &counts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < wsize; i++) {
            disps[i] = (i > 0) ? (disps[i - 1] + counts[i - 1]) : 0;
        }

        std::string result = "";

        if (shamcomm::world_rank() == 0) {
            u32 global_len = disps[wsize - 1] + counts[wsize - 1];
            result.resize(global_len);
        }

        shamcomm::mpi::Gatherv(
            send_vec.data(),
            send_vec.size(),
            MPI_CHAR,
            result.data(),
            counts.data(),
            disps.data(),
            MPI_CHAR,
            0,
            MPI_COMM_WORLD);

        recv_vec = result;
    }

} // namespace

void shamcomm::gather_str(const std::string &send_vec, std::string &recv_vec) {
    StackEntry stack_loc{};
    _internal_gather_str(send_vec, recv_vec);
}

void shamcomm::gather_basic_str(
    const std::basic_string<byte> &send_vec, std::basic_string<byte> &recv_vec) {
    StackEntry stack_loc{};
    _internal_gather_str(send_vec, recv_vec);
}

std::unordered_map<std::string, int>
shamcomm::string_histogram(const std::vector<std::string> &inputs, std::string delimiter) {
    std::string accum_loc = "";
    for (auto &s : inputs) {
        accum_loc += s + delimiter;
    }

    std::string recv = "";
    gather_str(accum_loc, recv);

    if (world_rank() == 0) {

        std::vector<std::string> splitted = shambase::split_str(recv, delimiter);

        std::unordered_map<std::string, int> histogram;

        for (size_t i = 0; i < splitted.size(); i++) {
            histogram[splitted[i]] += 1;
        }

        return histogram;
    }

    return {};
}

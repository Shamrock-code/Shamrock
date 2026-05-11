// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file string_histogram.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/checksum.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/string_histogram.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <unordered_map>
#include <string>
#include <vector>

std::unordered_map<std::string, int> shamalgs::collective::string_histogram(
    const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {

    if (hash_based) {
        std::vector<u64> fnv1a_in(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            fnv1a_in[i] = shambase::fnv1a_hash(inputs[i].data(), inputs[i].size());
        }
        std::vector<u64> fnv1a_recv;
        // shamalgs::collective::vector_allgatherv()(fnv1a_in, fnv1a_recv);
        std::vector<std::string> fnv1a_recv_str(fnv1a_recv.size());
        for (size_t i = 0; i < fnv1a_recv.size(); i++) {
            fnv1a_recv_str[i] = std::to_string(fnv1a_recv[i]);
        }
    }

    std::string accum_loc = "";
    for (auto &s : inputs) {
        accum_loc += s + delimiter;
    }

    std::string recv = "";
    gather_str(accum_loc, recv);

    if (shamcomm::world_rank() == 0) {

        std::vector<std::string> splitted = shambase::split_str(recv, delimiter);

        std::unordered_map<std::string, int> histogram;

        for (size_t i = 0; i < splitted.size(); i++) {
            histogram[splitted[i]] += 1;
        }

        return histogram;
    }

    return {};
}

std::unordered_map<std::string, int> shamalgs::collective::all_string_histogram(
    const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {
    std::string accum_loc = "";
    for (auto &s : inputs) {
        accum_loc += s + delimiter;
    }

    std::string recv = "";
    allgather_str(accum_loc, recv);

    std::vector<std::string> splitted = shambase::split_str(recv, delimiter);

    std::unordered_map<std::string, int> histogram;

    for (size_t i = 0; i < splitted.size(); i++) {
        histogram[splitted[i]] += 1;
    }

    return histogram;
}

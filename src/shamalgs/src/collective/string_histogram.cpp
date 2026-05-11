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
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/string_histogram.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace {
    inline std::vector<std::string> _gather_strings_internal(
        const std::vector<std::string> &inputs, const std::string &delimiter, bool is_allgather) {
        std::string accum_loc = "";
        for (auto &s : inputs) {
            accum_loc += s + delimiter;
        }
        std::string recv = "";

        if (is_allgather) {
            shamalgs::collective::allgather_str(accum_loc, recv);
            return shambase::split_str(recv, delimiter);
        }

        shamalgs::collective::gather_str(accum_loc, recv);
        if (shamcomm::world_rank() == 0) {
            return shambase::split_str(recv, delimiter);
        }

        return {};
    }

    std::unordered_map<std::string, int> _string_histogram_all_fetch(
        const std::vector<std::string> &inputs, std::string delimiter, bool is_allgather) {

        auto splitted = _gather_strings_internal(inputs, delimiter, is_allgather);

        if (is_allgather || shamcomm::world_rank() == 0) {
            std::unordered_map<std::string, int> histogram;
            for (size_t i = 0; i < splitted.size(); i++) {
                histogram[splitted[i]] += 1;
            }
            return histogram;
        }

        return {};
    }

    inline auto hash_inputs(const std::vector<std::string> &inputs) {
        std::vector<u64_2> fnv1a_in(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            fnv1a_in[i] = {
                shambase::fnv1a_hash(inputs[i].data(), inputs[i].size()), shamcomm::world_rank()};
        }
        return fnv1a_in;
    }

    struct CaseInfo {
        u64 min_rank_id;
        u64 count;
    };

    auto data_to_case_info(const std::vector<u64_2> &data) {
        std::unordered_map<u64, CaseInfo> hash_case_info = {};
        for (size_t i = 0; i < data.size(); i++) {
            auto hash = data[i].x();
            auto rank = data[i].y();

            if (hash_case_info.find(hash) == hash_case_info.end()) {
                hash_case_info[hash] = {rank, 1};
            } else {
                hash_case_info[hash].count += 1;
                hash_case_info[hash].min_rank_id = std::min(hash_case_info[hash].min_rank_id, rank);
            }
        }
        return hash_case_info;
    }

    std::unordered_map<std::string, int> _string_histogram_hash_fetch(
        const std::vector<std::string> &inputs, std::string delimiter, bool is_allgather) {

        // compute the hash of the inputs
        std::vector<u64_2> fnv1a_in = hash_inputs(inputs);

        // gather the hashes
        std::vector<u64_2> fnv1a_recv;
        shamalgs::collective::vector_allgatherv(fnv1a_in, fnv1a_recv, MPI_COMM_WORLD);

        // list all the unique hashes, the minimum rank id for the first occurrence and their count
        std::unordered_map<u64, CaseInfo> hash_case_info = data_to_case_info(fnv1a_recv);

        // restrict the inputs to the minimum rank id for the first occurrence
        std::vector<std::string> restricted_inputs = {};
        for (size_t i = 0; i < inputs.size(); i++) {
            auto hash = fnv1a_in[i].x();
            if (hash_case_info[hash].min_rank_id == shamcomm::world_rank()) {
                restricted_inputs.push_back(inputs[i]);
            }
        }

        auto histogram = _string_histogram_all_fetch(restricted_inputs, delimiter, is_allgather);

        // override the histogram from the hash case info
        for (auto &[word, cnt] : histogram) {
            auto fnv        = shambase::fnv1a_hash(word.data(), word.size());
            histogram[word] = hash_case_info[fnv].count;
        }

        return histogram;
    }

} // namespace

std::unordered_map<std::string, int> shamalgs::collective::string_histogram(
    const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {

    if (hash_based) {
        return _string_histogram_hash_fetch(inputs, delimiter, false);
    }

    return _string_histogram_all_fetch(inputs, delimiter, false);
}

std::unordered_map<std::string, int> shamalgs::collective::all_string_histogram(
    const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {

    if (hash_based) {
        return _string_histogram_hash_fetch(inputs, delimiter, true);
    }

    return _string_histogram_all_fetch(inputs, delimiter, true);
}

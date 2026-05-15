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
 * @file string_histogram.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief MPI string gather / allgather helpers (declarations; implementations in
 *        shamalgs/src/collective/gather_str.cpp).
 */

#include "shambase/aliases_int.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace shamalgs::collective {

    /**
     * @brief Constructs a histogram from a vector of strings, counting occurrences
     *        of each unique string.
     *
     * This function takes a vector of strings, concatenates them into a single
     * string using the specified delimiter, and then splits the concatenated
     * string back into individual strings. It then counts the occurrences of each
     * unique string and returns a histogram as an unordered map.
     *
     * @param inputs A vector of strings to process.
     * @param delimiter A string used to concatenate and split the inputs. Defaults
     *                  to a newline character.
     * @return An unordered map where keys are unique strings from the input and
     *         values are the counts of their occurrences. (valid only on rank 0)
     */
    std::unordered_map<std::string, int> string_histogram(
        const std::vector<std::string> &inputs, std::string delimiter, bool hash_based);

    /// same as string_histogram but with result return on every rank
    std::unordered_map<std::string, int> all_string_histogram(
        const std::vector<std::string> &inputs, std::string delimiter, bool hash_based);

} // namespace shamalgs::collective

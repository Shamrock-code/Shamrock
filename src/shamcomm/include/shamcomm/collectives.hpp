// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file collectives.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include <unordered_map>
#include <string>
#include <vector>

namespace shamcomm {

    /**
     * @brief Gathers a string from all nodes and store the result in a std::string
     *
     * This function gathers the string `send_vec` from all nodes and concatenate
     * the result in `recv_vec`. The result is ordered by the order of the nodes in
     * the communicator, i.e. the string is ordered by rank.
     *
     * \todo add fault tolerance
     *
     * @param send_vec The string to gather from all nodes
     * @param recv_vec The result of the gather. It will contain a concatenation
     *                of all the strings gathered from the nodes.
     */
    void gather_str(const std::string &send_vec, std::string &recv_vec);

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

    std::unordered_map<std::string, int>
    string_histogram(const std::vector<std::string> &inputs, std::string delimiter = "\n");

} // namespace shamcomm

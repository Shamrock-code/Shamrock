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
 * @file gather_str.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief MPI string gather / allgather helpers (declarations; implementations in
 *        shamalgs/src/collective/gather_str.cpp).
 */

#include "shambase/aliases_int.hpp"
#include <string>

namespace shamalgs::collective {

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

    /// same as gather_str but with std::basic_string
    void gather_basic_str(
        const std::basic_string<byte> &send_vec, std::basic_string<byte> &recv_vec);

    /**
     * @brief Allgathers a string from all nodes and concatenates it in a std::string
     *
     * This function gathers the string `send_vec` from all nodes and concatenates the
     * result in `recv_vec` on every rank. The result is ordered by the order of the
     * nodes in the communicator, i.e. the string is ordered by rank.
     */
    void allgather_str(const std::string &send_vec, std::string &recv_vec);

    /// same as allgather_str but with std::basic_string
    void allgather_basic_str(
        const std::basic_string<byte> &send_vec, std::basic_string<byte> &recv_vec);

} // namespace shamalgs::collective

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file collectives.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include <string>

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

} // namespace shamcomm

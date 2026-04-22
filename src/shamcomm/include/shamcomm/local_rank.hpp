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
 * @file local_rank.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Functions related to the MPI communicator
 *
 */

#include "shambase/aliases_int.hpp"
#include <optional>

namespace shamcomm {

    // fetch the current MPI rank within the current node
    std::optional<u32> node_local_rank();

    // returns true if node_local_rank() is not available, or if it matches main_local_rank_id
    bool is_main_node_rank(u32 main_local_rank_id = 0);

} // namespace shamcomm

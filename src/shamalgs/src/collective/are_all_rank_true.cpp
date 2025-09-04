// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file are_all_rank_true.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <shamcomm/mpi.hpp>

namespace shamalgs::collective {

    bool are_all_rank_true(bool input, MPI_Comm comm) {

        // Shamrock profiling / stack tracing entry
        [[maybe_unused]] StackEntry stack_loc{};

        if (shamcomm::world_size() == 1) {
            return input;
        }

        int tmp = input;
        int out = 0;

        shamcomm::mpi::Allreduce(&tmp, &out, 1, MPI_INT, MPI_LAND, comm);

        return out;
    }

} // namespace shamalgs::collective

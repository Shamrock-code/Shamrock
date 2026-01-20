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
 * @file are_all_rank_true.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Collective boolean reduction to check if all ranks have true as input
 *
 */

#include <shamcomm/mpi.hpp>

namespace shamalgs::collective {

    /// return true only if all ranks have true as input
    bool are_all_rank_true(bool input, MPI_Comm comm);

} // namespace shamalgs::collective

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
 * @file indexing.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"

namespace shamalgs::collective {

    struct ViewInfo {
        u64 total_byte_count;
        u64 head_offset;
    };

    inline ViewInfo fetch_view(u64 byte_count) {

        u64 scan_val;
        shamcomm::mpi::Exscan(
            &byte_count, &scan_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        if (shamcomm::world_rank() == 0) {
            scan_val = 0;
        }

        u64 sum_val;
        shamcomm::mpi::Allreduce(
            &byte_count, &sum_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        shamlog_debug_mpi_ln("fetch view", byte_count, "->", scan_val, "sum:", sum_val);

        return {sum_val, scan_val};
    }

    inline ViewInfo fetch_view_known_total(u64 byte_count, u64 total_byte) {

        u64 scan_val;
        shamcomm::mpi::Exscan(
            &byte_count, &scan_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        if (shamcomm::world_rank() == 0) {
            scan_val = 0;
        }

        shamlog_debug_mpi_ln("fetch view", byte_count, "->", scan_val, "sum:", total_byte);

        return {total_byte, scan_val};
    }

} // namespace shamalgs::collective

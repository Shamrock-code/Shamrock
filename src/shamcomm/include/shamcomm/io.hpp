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
 * @file io.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcomm/mpi.hpp"
#include <string>

namespace shamcomm {

    /**
     * @brief Open a MPI file and remove its content
     *
     * This function opens the MPI file `fname` with write-exclusive permission
     * and remove its content if the file already exists.
     *
     * @param[out] fh The MPI file handle to open
     * @param fname The name of the file to open
     */
    void open_reset_file(MPI_File &fh, const std::string &fname);

    /**
     * @brief Open a mpi file in read only mode
     */
    void open_read_only_file(MPI_File &fh, const std::string &fname);

} // namespace shamcomm

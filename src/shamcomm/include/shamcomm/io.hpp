// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file io.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
    void open_reset_file(MPI_File &fh, std::string fname);

    /**
     * @brief Open a mpi file in read only mode
     */
    void open_read_only_file(MPI_File &fh, std::string fname);

} // namespace shamcomm

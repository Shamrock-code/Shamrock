// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file io.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamcomm/io.hpp"
#include "shambase/exception.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"

namespace shamcomm {
    /**
     * @brief open a mpi file and remove its content
     *
     */
    void open_reset_file(MPI_File &fh, std::string fname) {

        int rc = MPI_File_open(
            MPI_COMM_WORLD,
            fname.c_str(),
            MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN,
            MPI_INFO_NULL,
            &fh);

        if (rc != MPI_SUCCESS) {

            if (shamcomm::world_rank() == 0) {
                MPICHECK(MPI_File_delete(fname.c_str(), MPI_INFO_NULL));
            }

            int rc = MPI_File_open(
                MPI_COMM_WORLD,
                fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN,
                MPI_INFO_NULL,
                &fh);

            if (rc != MPI_SUCCESS) {
                throw shambase::throw_with_loc<std::runtime_error>("cannot create file");
            }
        }
    }
} // namespace shamcomm
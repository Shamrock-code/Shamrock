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
     * @brief open a mpi file and remove its content
     *
     */
    void open_reset_file(MPI_File &fh, std::string fname);

} // namespace shamcomm
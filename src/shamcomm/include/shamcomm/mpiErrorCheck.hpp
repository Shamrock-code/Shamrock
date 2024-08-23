// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mpiErrorCheck.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Utility functions for MPI error checking
 *
 * This file contains a simple function to check the return code of MPI
 * calls and abort the program if an error occured. The function
 * prints an error message indicating where the error occured (using the
 * provided log string).
 *
 * The function `check_mpi_return` is the only function provided in this file.
 */

namespace shamcomm {

    /**
     * @brief Check a MPI return code
     *
     * This function checks the return code of an MPI call and aborts the
     * program if an error occured. The function prints an error message
     * indicating where the error occured (using the provided log string)
     *
     * @param ret The MPI return code to check
     * @param log A string indicating where the error occured
     */
    void check_mpi_return(int ret, const char *log);

} // namespace shamcomm

/**
 * @brief Shortcut macro to check MPI return codes
 *
 * This macro is a shortcut to avoid having to write
 * `shamcomm::check_mpi_return(mpicall, #mpicall)`
 * everywhere in the code.
 *
 * It expands to a call to `shamcomm::check_mpi_return` with
 * the MPI return code and the name of the MPI function as
 * arguments. This last argument is used to the MPI call that produced the error
 */
#define MPICHECK(mpicall) shamcomm::check_mpi_return(mpicall, #mpicall)

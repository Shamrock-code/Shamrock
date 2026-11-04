// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/MPIInitGuard.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"

NEW_TEST(Unittest, "shamcomm/MPIInitGuard", 1) {
    REQUIRE(shamcomm::is_mpi_initialized());

    {
        shamcomm::MPIInitGuard extra;
        REQUIRE(!extra.is_active());
        REQUIRE(shamcomm::is_mpi_initialized());

        extra.close();
        REQUIRE(!extra.is_active());
        REQUIRE(shamcomm::is_mpi_initialized());

        extra.close();
        REQUIRE(shamcomm::is_mpi_initialized());
    }

    REQUIRE(shamcomm::is_mpi_initialized());
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file worldInfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Use this header to include MPI properly
 *
 */

#include "shambase/aliases_int.hpp"

namespace shamcomm {

    /**
     * @brief the MPI world rank
     */
    const i32 world_rank();

    /**
     * @brief the MPI world size
     */
    const i32 world_size();

    void fetch_world_info();

} // namespace shamcomm
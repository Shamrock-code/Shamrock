// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file collectives.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include <string>

namespace shamcomm {

    /**
     * @brief 
     * \todo add fault tolerance
     * 
     * @param send_vec 
     * @param recv_vec 
     */
    void gather_str(const std::string &send_vec, std::string &recv_vec);

} // namespace shamcomm
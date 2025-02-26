// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file shamrock_smi.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include <functional>

namespace shamsys {

    void shamrock_smi() {
        if (!shamcomm::is_mpi_initialized()) {
            using namespace shamsys::instance;
            start_mpi(MPIInitInfo{opts::get_argc(), opts::get_argv()});
        }
    }
} // namespace shamsys

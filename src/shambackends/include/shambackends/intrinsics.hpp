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
 * @file intrinsics.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include "shambackends/intrisics/get_device_clock.hpp"
#include "shambackends/intrisics/get_sm_id.hpp"
#include <shambackends/sycl.hpp>

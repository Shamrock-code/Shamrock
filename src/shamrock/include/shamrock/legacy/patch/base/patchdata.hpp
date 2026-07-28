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
 * @file patchdata.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief header for PatchData related function and declaration
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <random>
#include <variant>
#include <vector>

/**
 * @brief generate dummy PatchDataLayer from a mersen twister
 *
 * @param eng the mersen twister
 * @return PatchDataLayer the generated PatchDataLayer
 */
shamrock::patch::PatchDataLayer patchdata_gen_dummy_data(
    const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &pdl_ptr, std::mt19937 &eng);

/**
 * @brief check if two PatchDataLayer content match
 *
 * @param p1
 * @param p2
 * @return true
 * @return false
 */
bool patch_data_check_match(
    shamrock::patch::PatchDataLayer &p1, shamrock::patch::PatchDataLayer &p2);

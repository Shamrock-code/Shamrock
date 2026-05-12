// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief implementation of PatchData related functions
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

shamrock::patch::PatchDataLayer patchdata_gen_dummy_data(
    const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &pdl_ptr, std::mt19937 &eng) {

    using namespace shamrock::patch;

    std::uniform_int_distribution<u64> distu64(1, 6000);

    u32 num_part = distu64(eng);

    PatchDataLayer pdat(pdl_ptr);

    pdat.for_each_field_any([&](auto &field) {
        field.gen_mock_data(num_part, eng);
    });

    return pdat;
}

bool patch_data_check_match(
    shamrock::patch::PatchDataLayer &p1, shamrock::patch::PatchDataLayer &p2) {

    return p1 == p2;
}

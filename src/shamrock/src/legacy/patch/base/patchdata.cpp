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
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

u64 patchdata_isend(
    shamrock::patch::PatchDataLayer &p,
    std::vector<PatchDataMpiRequest> &rq_lst,
    i32 rank_dest,
    i32 tag,
    MPI_Comm comm) {

    rq_lst.resize(rq_lst.size() + 1);
    PatchDataMpiRequest &ref = rq_lst[rq_lst.size() - 1];

    u64 total_data_transf = 0;

    p.for_each_field_any([&](auto &field) {
        using base_t = typename std::remove_reference<decltype(field)>::type::Field_type;
        total_data_transf
            += patchdata_field::isend(field, ref.get_field_list<base_t>(), rank_dest, tag, comm);
    });

    return total_data_transf;
}

u64 patchdata_irecv_probe(
    shamrock::patch::PatchDataLayer &pdat,
    std::vector<PatchDataMpiRequest> &rq_lst,
    i32 rank_source,
    i32 tag,
    MPI_Comm comm) {

    rq_lst.resize(rq_lst.size() + 1);
    auto &ref = rq_lst[rq_lst.size() - 1];

    u64 total_data_transf = 0;

    pdat.for_each_field_any([&](auto &field) {
        using base_t = typename std::remove_reference<decltype(field)>::type::Field_type;
        total_data_transf += patchdata_field::irecv_probe(
            field, ref.get_field_list<base_t>(), rank_source, tag, comm);
    });

    return total_data_transf;
}

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

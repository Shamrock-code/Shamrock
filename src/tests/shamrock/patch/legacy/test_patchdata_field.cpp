// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/patch/utility/patch_field.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

const std::string base = "core/patch/base/patchdata_field";

/**
 *   Checking PatchDataField status after constructor
 */
TestStart(Unittest, base + ":constructor", patch_data_field_constructor, 1) {

    PatchDataField<f32> d_check("test", 1);

    REQUIRE_NAMED("name field match", d_check.get_name() == "test");
    REQUIRE_NAMED("nvar field match", d_check.get_nvar() == 1);
    REQUIRE_NAMED("buffer allocated but empty", d_check.get_buf().is_empty());

    REQUIRE_NAMED("is new field empty", d_check.get_val_cnt() == 0);
    REQUIRE_NAMED("is new field empty", d_check.get_obj_cnt() == 0);
}

TestStart(Unittest, base + ":patch_data_field_check_match", patch_data_field_check_match, 1) {
    std::mt19937 eng(0x1111);

    PatchDataField<f32> d_check("test", 1);
    d_check.gen_mock_data(10000, eng);

    REQUIRE_NAMED("reflexivity", d_check.check_field_match(d_check));
}

TestStart(Unittest, base + ":isend_irecv_f32", isend_irecv_f32, 2) {
    std::mt19937 eng(0x1111);

    PatchDataField<f32> d1_check("test", 1);
    PatchDataField<f32> d2_check("test", 1);

    std::uniform_int_distribution<u64> distu64(1, 6000);

    d1_check.gen_mock_data(distu64(eng), eng);
    d2_check.gen_mock_data(distu64(eng), eng);

    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32>> rq_lst;
    PatchDataField<f32> recv_d("test", 1);

    if (shamcomm::world_rank() == 0) {
        patchdata_field::isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_field::irecv_probe(recv_d, rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if (shamcomm::world_rank() == 1) {
        patchdata_field::isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_field::irecv_probe(recv_d, rq_lst, 0, 0, MPI_COMM_WORLD);
    }

    std::cout << "request len : [" << shamcomm::world_rank() << "] " << rq_lst.size() << std::endl;

    patchdata_field::waitall(rq_lst);

    if (shamcomm::world_rank() == 0) {
        PatchDataField<f32> &p1 = recv_d;
        PatchDataField<f32> &p2 = d2_check;
        REQUIRE_NAMED("recv_d == d1_check (f32)", p1.check_field_match(p2));
    }

    if (shamcomm::world_rank() == 1) {
        PatchDataField<f32> &p1 = recv_d;
        PatchDataField<f32> &p2 = d1_check;
        REQUIRE_NAMED("recv_d == d1_check (f32)", p1.check_field_match(p2));
    }
}

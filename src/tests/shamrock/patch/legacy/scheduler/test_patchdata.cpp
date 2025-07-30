// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtest/shamtest.hpp"
#include <random>
#include <vector>

/*
Test_start("patchdata::", sync_patchdata_layout, -1) {

    if (shamcomm::world_rank() == 0) {
        patchdata_layout::set(1, 0, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);

    Test_assert("sync nVarpos_s",patchdata_layout::nVarpos_s == 1);
    Test_assert("sync nVarpos_d",patchdata_layout::nVarpos_d == 0);
    Test_assert("sync nVarU1_s ",patchdata_layout::nVarU1_s  == 4);
    Test_assert("sync nVarU1_d ",patchdata_layout::nVarU1_d  == 6);
    Test_assert("sync nVarU3_s ",patchdata_layout::nVarU3_s  == 2);
    Test_assert("sync nVarU3_d ",patchdata_layout::nVarU3_d  == 1);

}
*/

TestStart(Unittest, "patchdata::", send_recv_patchdata, 2) {

    using namespace shamrock::patch;

    std::mt19937 eng(0x1111);

    PatchDataLayout pdl;

    pdl.add_field<f32_3>("xyz", 1);

    pdl.add_field<f64_8>("test", 2);

    PatchData d1_check = patchdata_gen_dummy_data(pdl, eng);
    PatchData d2_check = patchdata_gen_dummy_data(pdl, eng);

    std::vector<PatchDataMpiRequest> rq_lst;
    PatchData recv_d(pdl);

    if (shamcomm::world_rank() == 0) {
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_irecv_probe(recv_d, rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if (shamcomm::world_rank() == 1) {
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_irecv_probe(recv_d, rq_lst, 0, 0, MPI_COMM_WORLD);
    }

    waitall_pdat_mpi_rq(rq_lst);

    if (shamcomm::world_rank() == 0) {
        REQUIRE_NAMED("recv_d == d2_check", patch_data_check_match(recv_d, d2_check));
    }

    if (shamcomm::world_rank() == 1) {
        REQUIRE_NAMED("recv_d == d1_check", patch_data_check_match(recv_d, d1_check));
    }
}

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
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

TestStart(Unittest, "patchdata.cpp/patch_data_check_match", patch_data_check_match, 1) {
    std::mt19937 eng(0x1111);

    using namespace shamrock::patch;

    PatchDataLayout pdl;

    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);

    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3'", 1);
    pdl.add_field<f32_3>("f32_3''", 1);

    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 1);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);

    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);

    PatchDataLayer d_check = patchdata_gen_dummy_data(pdl, eng);

    REQUIRE_NAMED("reflexivity", patch_data_check_match(d_check, d_check));
}

/*

TestStart(Unittest, "patchdata.cpp/isend_irecv",patch_data_isend_irecv, 2){
    std::mt19937 eng(0x1111);

    using namespace shamrock::patch;


    PatchDataLayout pdl;

    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);

    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3'", 2);
    pdl.add_field<f32_3>("f32_3''", 1);

    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 1);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);

    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);





    PatchData d1_check = patchdata_gen_dummy_data (pdl,eng);
    PatchData d2_check = patchdata_gen_dummy_data (pdl,eng);



    std::vector<PatchDataMpiRequest> rq_lst;
    PatchData recv_d(pdl);


    if(shamcomm::world_rank() == 0){
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        patchdata_irecv_probe(recv_d,rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        patchdata_irecv_probe(recv_d,rq_lst, 0, 0, MPI_COMM_WORLD);
    }




    //std::cout << "request len : [" << shamcomm::world_rank() << "] " << rq_lst.size() <<
std::endl;

    waitall_pdat_mpi_rq(rq_lst);


    if(shamcomm::world_rank() == 0){
        shamtest::asserts().assert_bool("recv_d == d2_check", patch_data_check_match(recv_d,
d2_check));
    }

    if(shamcomm::world_rank() == 1){
        shamtest::asserts().assert_bool("recv_d == d1_check", patch_data_check_match(recv_d,
d1_check));
    }




}

*/

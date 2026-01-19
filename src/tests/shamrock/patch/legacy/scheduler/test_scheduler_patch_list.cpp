// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include "test_patch_utils.hpp"

#if false

//TODO add other test to check "make_fake_patch_list"
Test_start("SchedulerPatchList::fake_patch_list_gen()::",patch_are_cubes,-1){

    std::vector<Patch> test_vec = make_fake_patch_list(200, 10);


    bool all_pcube = true;

    for(Patch & p : test_vec){

        u64 dx = p.x_max - p.x_min;
        u64 dy = p.y_max - p.y_min;
        u64 dz = p.z_max - p.z_min;

        all_pcube = all_pcube && ( dx == dy && dy == dz);
    }

    Test_assert("all patch are cubes", all_pcube);

}


Test_start("SchedulerPatchList::fake_patch_list_gen()::",no_patch_intersection,-1){

    std::vector<Patch> test_vec = make_fake_patch_list(200, 10);

    bool no_p_intersect = true;

    for(Patch & p1 : test_vec){
        for(Patch & p2 : test_vec){

            if(p1 == p2) continue;

            no_p_intersect = no_p_intersect && ( !BBAA::intersect_not_null_cella_b<u32_3>(
                {p1.x_min,p1.y_min,p1.z_min},{p1.x_max,p1.y_max,p1.z_max},
                {p2.x_min,p2.y_min,p2.z_min},{p2.x_max,p2.y_max,p2.z_max}
                ));

        }
    }

    Test_assert("no intersect", no_p_intersect);

}


//TODO use "make_fake_patch_list" instead of the old utility
Test_start("SchedulerPatchList::",build_select_corectness,-1){
    patch::create_MPI_patch_type();

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);


    SchedulerPatchList patch_list = SchedulerPatchList();
    for(const Patch &p : check_vec){
        patch_list.global.push_back(p);
    }
    auto res = patch_list.build_local();


    //check corectness of local patch list
    bool corect_size = patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", patch_list.local[i] == local_check_vec[i]);
        }
    }


    patch_list.global.clear();
    patch_list.build_global();


    corect_size = patch_list.global.size() == check_vec.size();
    Test_assert("corect size for global patch", corect_size);
    for(u32 i = 0 ; i < patch_list.global.size(); i++){
        if(corect_size){
            Test_assert("corect patch", patch_list.global[i] == check_vec[i]);
        }
    }



    patch::free_MPI_patch_type();
}

#endif

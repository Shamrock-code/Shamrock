// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shambase/sycl_utils.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase::parralel_for", test_par_for_1d, 1){
    u32 len = 10000;
    sycl::buffer<u64> buf (len);

    shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
        sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parralel_for(cgh, len, "test 1d par for",[=](u64 id){
            acc[id] = id;
        });
    });

    bool correct = true;
    {
        sycl::host_accessor acc {buf,sycl::read_only};
        for(u32 x = 0; x < len; x++){
            if(acc[x] != x){correct = false;}
        }
    }
    _Assert(correct);
}

TestStart(Unittest, "shambase::parralel_for", test_par_for_2d, 1){
    u64 len_x = 1000;
    u64 len_y = 1000;
    sycl::buffer<u64,2> buf (sycl::range<2>{len_x,len_y});

    shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
        sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parralel_for_2d(cgh, len_x,len_y, "test 2d par for",[=](u64 id_x, u64 id_y){
            acc[{len_x,len_y}] = id_x + len_x*id_y;
        });
    });

    bool correct = true;
    {
        sycl::host_accessor acc {buf,sycl::read_only};
        for(u32 x = 0; x < len_x; x++){
            for(u32 y = 0; y < len_y; y++){
                if(acc[{x,y}] != x + len_x*y){correct = false;}
            }
        }
    }
    _Assert(correct);
}

TestStart(Unittest, "shambase::parralel_for", test_par_for_3d, 1){
    u64 len_x = 100;
    u64 len_y = 100;
    u64 len_z = 100;
    sycl::buffer<u64,3> buf (sycl::range<3>{len_x,len_y,len_z});

    shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
        sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parralel_for_3d(cgh, len_x,len_y,len_z, "test 2d par for",[=](u64 id_x, u64 id_y,u64 id_z){
            acc[{len_x,len_y,len_z}] = id_x + len_x*id_y+ len_x*len_y*id_z;
        });
    });

    bool correct = true;
    {
        sycl::host_accessor acc {buf,sycl::read_only};
        for(u32 x = 0; x < len_x; x++){
            for(u32 y = 0; y < len_y; y++){
                for(u32 z = 0; z < len_z; z++){
                    if(acc[{x,y,z}] != x + len_x*y+ len_x*len_y*z){correct = false;}
                }
            }
        }
    }
    _Assert(correct);
}

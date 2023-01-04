// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"
#include "shamtest/shamtest.hpp"

#include "shamrock/sfc/morton.hpp"
#include "shamrock/tree/kernels/morton_kernels.hpp"
#include <memory>
#include <vector>



#if false



Test_start("morton::",min_max_value,1){


    u32 m_0_32 = morton_3d::coord_to_morton<u32,f32>(0, 0, 0);
    u32 m_max_32 = morton_3d::coord_to_morton<u32,f32>(1, 1, 1);

    u64 m_0_64 = morton_3d::coord_to_morton<u64,f32>(0, 0, 0);
    u64 m_max_64 = morton_3d::coord_to_morton<u64,f32>(1, 1, 1);

    
    Test_assert("min morton 64 == b0", m_0_64 == 0x0);    
    Test_assert("max morton 64 == b63x1", m_max_64 == 0x7fffffffffffffff);
    
    Test_assert("min morton 32 == b0x0", m_0_32 == 0x0);    
    Test_assert("max morton 32 == b30x1", m_max_32 == 0x3fffffff);
    

}

Test_start("tree::kernels::",morton_kernels,1){

    std::vector<f32_3> xyz_32 {{0,0,0},{1,1,1}};
    std::vector<u32>   morton_32(2);

    {

        std::unique_ptr<sycl::buffer<f32_3>> buf_xyz    = std::make_unique<sycl::buffer<f32_3>>(xyz_32.data(),xyz_32.size());
        std::unique_ptr<sycl::buffer<u32>>   buf_morton = std::make_unique<sycl::buffer<u32>>(morton_32.data(),morton_32.size());

        sycl_xyz_to_morton<u32,f32_3>(sycl_handler::get_compute_queue(), 2, buf_xyz, f32_3{0,0,0}, f32_3{1,1,1}, buf_morton);

    }

    Test_assert("min morton 32 == b0x0", morton_32[0] == 0x0);    
    Test_assert("max morton 32 == b30x1", morton_32[1] == 0x3fffffff);


    std::vector<f64_3> xyz_64 {{0,0,0},{1,1,1}};
    std::vector<u64>   morton_64(2);

    {

        std::unique_ptr<sycl::buffer<f64_3>> buf_xyz    = std::make_unique<sycl::buffer<f64_3>>(xyz_64.data(),xyz_64.size());
        std::unique_ptr<sycl::buffer<u64>>   buf_morton = std::make_unique<sycl::buffer<u64>>(morton_64.data(),morton_64.size());

        sycl_xyz_to_morton<u64,f64_3>(sycl_handler::get_compute_queue(), 2, buf_xyz, f64_3{0,0,0}, f64_3{1,1,1}, buf_morton);

    }

    Test_assert("min morton 64 == b0", morton_64[0] == 0x0);    
    Test_assert("max morton 64 == b63x1", morton_64[1] == 0x7fffffffffffffff);
    

}

#endif
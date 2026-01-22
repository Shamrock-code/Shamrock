// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/sfc/morton.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

#if false

Test_start("tree::kernels::",morton_kernels,1){

    std::vector<f32_3> xyz_32 {{0,0,0},{1,1,1}};
    std::vector<u32>   morton_32(2);

    {

        std::unique_ptr<sycl::buffer<f32_3>> buf_xyz    = std::make_unique<sycl::buffer<f32_3>>(xyz_32.data(),xyz_32.size());
        std::unique_ptr<sycl::buffer<u32>>   buf_morton = std::make_unique<sycl::buffer<u32>>(morton_32.data(),morton_32.size());

        sycl_xyz_to_morton<u32,f32_3>(shamsys::instance::get_compute_queue(), 2, buf_xyz, f32_3{0,0,0}, f32_3{1,1,1}, buf_morton);

    }

    Test_assert("min morton 32 == b0x0", morton_32[0] == 0x0);
    Test_assert("max morton 32 == b30x1", morton_32[1] == 0x3fffffff);


    std::vector<f64_3> xyz_64 {{0,0,0},{1,1,1}};
    std::vector<u64>   morton_64(2);

    {

        std::unique_ptr<sycl::buffer<f64_3>> buf_xyz    = std::make_unique<sycl::buffer<f64_3>>(xyz_64.data(),xyz_64.size());
        std::unique_ptr<sycl::buffer<u64>>   buf_morton = std::make_unique<sycl::buffer<u64>>(morton_64.data(),morton_64.size());

        sycl_xyz_to_morton<u64,f64_3>(shamsys::instance::get_compute_queue(), 2, buf_xyz, f64_3{0,0,0}, f64_3{1,1,1}, buf_morton);

    }

    Test_assert("min morton 64 == b0", morton_64[0] == 0x0);
    Test_assert("max morton 64 == b63x1", morton_64[1] == 0x7fffffffffffffff);


}

#endif

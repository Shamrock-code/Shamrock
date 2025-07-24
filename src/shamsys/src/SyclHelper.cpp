// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SyclHelper.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamsys/SyclHelper.hpp"

namespace shamsys::syclhelper::mock {

    template<class T>
    sycl::buffer<T> mock_buffer(u32 len, std::mt19937 &eng) {
        std::uniform_real_distribution<f64> distval(-1.0F, 1.0F);

        sycl::buffer<T> buf(len);

        {
            sycl::host_accessor acc{buf};
            for (u32 i = 0; i < len; i++) {
                acc[i] = next_obj<T>(eng, distval);
            }
        }

        return std::move(buf);
    }

    template sycl::buffer<f32> mock_buffer<f32>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f32_2> mock_buffer<f32_2>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f32_3> mock_buffer<f32_3>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f32_4> mock_buffer<f32_4>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f32_8> mock_buffer<f32_8>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f32_16> mock_buffer<f32_16>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64> mock_buffer<f64>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64_2> mock_buffer<f64_2>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64_3> mock_buffer<f64_3>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64_4> mock_buffer<f64_4>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64_8> mock_buffer<f64_8>(u32 len, std::mt19937 &eng);
    template sycl::buffer<f64_16> mock_buffer<f64_16>(u32 len, std::mt19937 &eng);
    template sycl::buffer<u32> mock_buffer<u32>(u32 len, std::mt19937 &eng);
    template sycl::buffer<u64> mock_buffer<u64>(u32 len, std::mt19937 &eng);

} // namespace shamsys::syclhelper::mock

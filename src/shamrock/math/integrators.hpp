// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file integrators.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock {
    namespace integrators {

        template<class flt, class T>
        void forward_euler(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &buf_val,
            sham::DeviceBuffer<T> &buf_der,
            sycl::range<1> elem_range,
            flt dt);

        template<class flt, class T>
        void leapfrog_corrector(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &buf_val,
            sham::DeviceBuffer<T> &buf_der,
            sham::DeviceBuffer<T> &buf_der_old,
            sham::DeviceBuffer<flt> &buf_eps_sq,
            sycl::range<1> elem_range,
            flt hdt);
    } // namespace integrators

    namespace utilities {

        template<class T>
        void sycl_position_modulo(
            sycl::queue &queue,
            sycl::buffer<T> &buf_xyz,
            sycl::range<1> elem_range,
            std::pair<T, T> box);

        template<class T>
        void sycl_position_sheared_modulo(
            sycl::queue &queue,
            sycl::buffer<T> &buf_xyz,
            sycl::buffer<T> &buf_vxyz,
            sycl::range<1> elem_range,
            std::pair<T, T> box,
            i32_3 shear_base,
            i32_3 shear_dir,
            shambase::VecComponent<T> shear_value,
            shambase::VecComponent<T> shear_speed);

        template<class T>
        void swap_fields(sycl::queue &queue, sycl::buffer<T> &b1, sycl::buffer<T> &b2, u32 cnt);
    } // namespace utilities

} // namespace shamrock

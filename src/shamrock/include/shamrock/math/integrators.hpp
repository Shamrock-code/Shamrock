// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file integrators.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock {
    namespace integrators {

        /**
         * @brief Perform forward Euler integration step
         *
         * @tparam flt Floating point type for time step
         * @tparam T Value type
         * @param queue Device queue to submit operation
         * @param buf_val Values to integrate
         * @param buf_der Derivatives
         * @param elem_range Number of elements to process
         * @param dt Time step
         */
        template<class flt, class T>
        void forward_euler(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &buf_val,
            sham::DeviceBuffer<T> &buf_der,
            sycl::range<1> elem_range,
            flt dt);

        /**
         * @brief Perform leapfrog corrector step with adaptive softening
         *
         * @tparam flt Floating point type for time step
         * @tparam T Value type
         * @param queue Device queue to submit operation
         * @param buf_val Values to integrate
         * @param buf_der Current derivatives
         * @param buf_der_old Previous derivatives
         * @param buf_eps_sq Squared softening parameter per element
         * @param elem_range Number of elements to process
         * @param hdt Half time step
         */
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

        /**
         * @brief Apply periodic boundary conditions to positions
         *
         * @tparam T Vector type for positions
         * @param queue Device queue to submit operation
         * @param buf_xyz Position buffer
         * @param elem_range Number of elements to process
         * @param box Box bounds (min, max)
         */
        template<class T>
        void sycl_position_modulo(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &buf_xyz,
            sycl::range<1> elem_range,
            std::pair<T, T> box);

        /**
         * @brief Apply periodic boundary conditions with shearing
         *
         * @tparam T Vector type for positions and velocities
         * @param queue Device queue to submit operation
         * @param buf_xyz Position buffer
         * @param buf_vxyz Velocity buffer
         * @param elem_range Number of elements to process
         * @param box Box bounds (min, max)
         * @param shear_base Direction perpendicular to shear plane
         * @param shear_dir Shear direction
         * @param shear_value Shear displacement per box crossing
         * @param shear_speed Velocity correction per box crossing
         */
        template<class T>
        void sycl_position_sheared_modulo(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &buf_xyz,
            sham::DeviceBuffer<T> &buf_vxyz,
            sycl::range<1> elem_range,
            std::pair<T, T> box,
            i32_3 shear_base,
            i32_3 shear_dir,
            shambase::VecComponent<T> shear_value,
            shambase::VecComponent<T> shear_speed);

        /**
         * @brief Swap contents of two buffers
         *
         * @tparam T Buffer element type
         * @param queue Device queue to submit operation
         * @param b1 First buffer
         * @param b2 Second buffer
         * @param cnt Number of elements to swap
         */
        template<class T>
        void swap_fields(
            sham::DeviceQueue &queue,
            sham::DeviceBuffer<T> &b1,
            sham::DeviceBuffer<T> &b2,
            u32 cnt);
    } // namespace utilities

} // namespace shamrock

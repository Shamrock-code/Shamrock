// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file integrators.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 * \todo move formula to shammath
 */

#include "shambase/exception.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamrock/math/integrators.hpp"
#include <algorithm>

namespace integ = shamrock::integrators;
namespace util  = shamrock::utilities;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class flt, class T>
void integ::forward_euler(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<T> &buf_val,
    sham::DeviceBuffer<T> &buf_der,
    sycl::range<1> elem_range,
    flt dt) {

    sham::EventList depends_list;

    auto acc_u  = buf_val.get_write_access(depends_list);
    auto acc_du = buf_der.get_read_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid     = (u32) item.get_id();
            acc_u[item] = acc_u[item] + (dt) *acc_du[item];
        });
    });

    buf_val.complete_event_state(e);
    buf_der.complete_event_state(e);
}

#ifndef DOXYGEN
template void integ::forward_euler(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32_3> &buf_val,
    sham::DeviceBuffer<f32_3> &buf_der,
    sycl::range<1> elem_range,
    f32 dt);

template void integ::forward_euler(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32> &buf_val,
    sham::DeviceBuffer<f32> &buf_der,
    sycl::range<1> elem_range,
    f32 dt);

template void integ::forward_euler(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64_3> &buf_val,
    sham::DeviceBuffer<f64_3> &buf_der,
    sycl::range<1> elem_range,
    f64 dt);

template void integ::forward_euler(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64> &buf_val,
    sham::DeviceBuffer<f64> &buf_der,
    sycl::range<1> elem_range,
    f64 dt);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class flt, class T>
void integ::leapfrog_corrector(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<T> &buf_val,
    sham::DeviceBuffer<T> &buf_der,
    sham::DeviceBuffer<T> &buf_der_old,
    sham::DeviceBuffer<flt> &buf_eps_sq,
    sycl::range<1> elem_range,
    flt hdt) {

    sham::EventList depends_list;

    auto acc_u          = buf_val.get_write_access(depends_list);
    auto acc_du         = buf_der.get_read_access(depends_list);
    auto acc_du_old     = buf_der_old.get_read_access(depends_list);
    auto acc_epsilon_sq = buf_eps_sq.get_write_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id();

            T incr = (hdt) * (acc_du[item] - acc_du_old[item]);

            acc_u[item]          = acc_u[item] + incr;
            acc_epsilon_sq[item] = sycl::dot(incr, incr);
        });
    });

    buf_val.complete_event_state(e);
    buf_der.complete_event_state(e);
    buf_der_old.complete_event_state(e);
    buf_eps_sq.complete_event_state(e);
}

#ifndef DOXYGEN
template void integ::leapfrog_corrector(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32_3> &buf_val,
    sham::DeviceBuffer<f32_3> &buf_der,
    sham::DeviceBuffer<f32_3> &buf_der_old,
    sham::DeviceBuffer<f32> &buf_eps_sq,
    sycl::range<1> elem_range,
    f32 hdt);

template void integ::leapfrog_corrector(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32> &buf_val,
    sham::DeviceBuffer<f32> &buf_der,
    sham::DeviceBuffer<f32> &buf_der_old,
    sham::DeviceBuffer<f32> &buf_eps_sq,
    sycl::range<1> elem_range,
    f32 hdt);

template void integ::leapfrog_corrector(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64_3> &buf_val,
    sham::DeviceBuffer<f64_3> &buf_der,
    sham::DeviceBuffer<f64_3> &buf_der_old,
    sham::DeviceBuffer<f64> &buf_eps_sq,
    sycl::range<1> elem_range,
    f64 hdt);

template void integ::leapfrog_corrector(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64> &buf_val,
    sham::DeviceBuffer<f64> &buf_der,
    sham::DeviceBuffer<f64> &buf_der_old,
    sham::DeviceBuffer<f64> &buf_eps_sq,
    sycl::range<1> elem_range,
    f64 hdt);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::sycl_position_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<T> &buf_xyz,
    sycl::range<1> elem_range,
    std::pair<T, T> box) {

    sham::EventList depends_list;
    auto xyz = buf_xyz.get_write_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        T box_min = std::get<0>(box);
        T box_max = std::get<1>(box);
        T delt    = box_max - box_min;

        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id();

            T r = xyz[gid] - box_min;

            r = sycl::fmod(r, delt);
            r += delt;
            r = sycl::fmod(r, delt);
            r += box_min;

            xyz[gid] = r;
        });
    });

    buf_xyz.complete_event_state(e);
}

#ifndef DOXYGEN
template void util::sycl_position_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32_3> &buf_xyz,
    sycl::range<1> elem_range,
    std::pair<f32_3, f32_3> box);

template void util::sycl_position_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64_3> &buf_xyz,
    sycl::range<1> elem_range,
    std::pair<f64_3, f64_3> box);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::sycl_position_sheared_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<T> &buf_xyz,
    sham::DeviceBuffer<T> &buf_vxyz,
    sycl::range<1> elem_range,
    std::pair<T, T> box,
    i32_3 shear_base,
    i32_3 shear_dir,
    shambase::VecComponent<T> shear_value,
    shambase::VecComponent<T> shear_speed) {

    sham::EventList depends_list;
    auto xyz  = buf_xyz.get_write_access(depends_list);
    auto vxyz = buf_vxyz.get_write_access(depends_list);

    auto e
        = queue.submit(depends_list, [&, shear_base, shear_value, shear_speed](sycl::handler &cgh) {
              T box_min = std::get<0>(box);
              T box_max = std::get<1>(box);
              T delt    = box_max - box_min;

              cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
                  u32 gid = (u32) item.get_id();

                  T r = xyz[gid] - box_min;

                  T roff = r / delt;

                  // T dn = sycl::trunc(roff);

                  // auto d = sycl::dot(dn,shear_base.convert<shambase::VecComponent<T>>());

                  //*
                  auto cnt_per = [](shambase::VecComponent<T> v) -> int {
                      return (v >= 0) ? int(v) : (int(v) - 1);
                  };

                  i32 xoff = cnt_per(roff.x());
                  i32 yoff = cnt_per(roff.y());
                  i32 zoff = cnt_per(roff.z());

                  i32 dx = xoff * shear_base.x();
                  i32 dy = yoff * shear_base.y();
                  i32 dz = zoff * shear_base.z();

                  i32 d = dx + dy + dz;
                  //*/

                  T shift
                      = {(d * shear_dir.x()) * shear_value,
                         (d * shear_dir.y()) * shear_value,
                         (d * shear_dir.z()) * shear_value};

                  T shift_speed
                      = {(d * shear_dir.x()) * shear_speed,
                         (d * shear_dir.y()) * shear_speed,
                         (d * shear_dir.z()) * shear_speed};

                  vxyz[gid] -= shift_speed;
                  r -= shift;

                  r = sycl::fmod(r, delt);
                  r += delt;
                  r = sycl::fmod(r, delt);
                  r += box_min;

                  xyz[gid] = r;
              });
          });

    buf_xyz.complete_event_state(e);
    buf_vxyz.complete_event_state(e);
}

#ifndef DOXYGEN
template void util::sycl_position_sheared_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32_3> &buf_xyz,
    sham::DeviceBuffer<f32_3> &buf_vxyz,
    sycl::range<1> elem_range,
    std::pair<f32_3, f32_3> box,
    i32_3 shear_base,
    i32_3 shear_dir,
    f32 shear_value,
    f32 shear_speed);

template void util::sycl_position_sheared_modulo(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64_3> &buf_xyz,
    sham::DeviceBuffer<f64_3> &buf_vxyz,
    sycl::range<1> elem_range,
    std::pair<f64_3, f64_3> box,
    i32_3 shear_base,
    i32_3 shear_dir,
    f64 shear_value,
    f64 shear_speed);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::swap_fields(
    sham::DeviceQueue &queue, sham::DeviceBuffer<T> &b1, sham::DeviceBuffer<T> &b2, u32 cnt) {

    sham::EventList depends_list;
    auto acc1 = b1.get_write_access(depends_list);
    auto acc2 = b2.get_write_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{cnt}, [=](sycl::item<1> item) {
            T v1 = acc1[item];
            T v2 = acc2[item];

            acc1[item] = v2;
            acc2[item] = v1;
        });
    });

    b1.complete_event_state(e);
    b2.complete_event_state(e);
}

#ifndef DOXYGEN
template void util::swap_fields(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f32_3> &b1,
    sham::DeviceBuffer<f32_3> &b2,
    u32 cnt);

template void util::swap_fields(
    sham::DeviceQueue &queue, sham::DeviceBuffer<f32> &b1, sham::DeviceBuffer<f32> &b2, u32 cnt);

template void util::swap_fields(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<f64_3> &b1,
    sham::DeviceBuffer<f64_3> &b2,
    u32 cnt);

template void util::swap_fields(
    sham::DeviceQueue &queue, sham::DeviceBuffer<f64> &b1, sham::DeviceBuffer<f64> &b2, u32 cnt);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

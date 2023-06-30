// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/math/integrators.hpp"
#include "shambase/exception.hpp"
#include "shambase/sycl_utils.hpp"
#include <algorithm>

namespace integ = shamrock::integrators;
namespace util  = shamrock::utilities;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class flt, class T>
void integ::forward_euler(sycl::queue &queue,
                          sycl::buffer<T> &buf_val,
                          sycl::buffer<T> &buf_der,
                          sycl::range<1> elem_range,
                          flt dt) {

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor acc_u{buf_val, cgh, sycl::read_write};
        sycl::accessor acc_du{buf_der, cgh, sycl::read_only};

        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid     = (u32)item.get_id();
            acc_u[item] = acc_u[item] + (dt)*acc_du[item];
        });
    });
}

template void integ::forward_euler(sycl::queue &queue,
                                   sycl::buffer<f32_3> &buf_val,
                                   sycl::buffer<f32_3> &buf_der,
                                   sycl::range<1> elem_range,
                                   f32 dt);

template void integ::forward_euler(sycl::queue &queue,
                                   sycl::buffer<f32> &buf_val,
                                   sycl::buffer<f32> &buf_der,
                                   sycl::range<1> elem_range,
                                   f32 dt);

template void integ::forward_euler(sycl::queue &queue,
                                   sycl::buffer<f64_3> &buf_val,
                                   sycl::buffer<f64_3> &buf_der,
                                   sycl::range<1> elem_range,
                                   f64 dt);

template void integ::forward_euler(sycl::queue &queue,
                                   sycl::buffer<f64> &buf_val,
                                   sycl::buffer<f64> &buf_der,
                                   sycl::range<1> elem_range,
                                   f64 dt);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class flt, class T>
void integ::leapfrog_corrector(sycl::queue &queue,
                               sycl::buffer<T> &buf_val,
                               sycl::buffer<T> &buf_der,
                               sycl::buffer<T> &buf_der_old,
                               sycl::buffer<flt> &buf_eps_sq,
                               sycl::range<1> elem_range,
                               flt hdt) {

    shambase::check_buffer_size(buf_val, elem_range.size());
    shambase::check_buffer_size(buf_der, elem_range.size());
    shambase::check_buffer_size(buf_der_old, elem_range.size());
    shambase::check_buffer_size(buf_eps_sq, elem_range.size());

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor acc_u{buf_val, cgh, sycl::read_write};
        sycl::accessor acc_du{buf_der, cgh, sycl::read_only};
        sycl::accessor acc_du_old{buf_der_old, cgh, sycl::read_only};
        sycl::accessor acc_epsilon_sq{buf_eps_sq, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            T incr = (hdt) * (acc_du[item] - acc_du_old[item]);

            acc_u[item]          = acc_u[item] + incr;
            acc_epsilon_sq[item] = sycl::dot(incr, incr);
        });
    });
}

template void integ::leapfrog_corrector(sycl::queue &queue,
                                        sycl::buffer<f32_3> &buf_val,
                                        sycl::buffer<f32_3> &buf_der,
                                        sycl::buffer<f32_3> &buf_der_old,
                                        sycl::buffer<f32> &buf_eps_sq,
                                        sycl::range<1> elem_range,
                                        f32 hdt);

template void integ::leapfrog_corrector(sycl::queue &queue,
                                        sycl::buffer<f32> &buf_val,
                                        sycl::buffer<f32> &buf_der,
                                        sycl::buffer<f32> &buf_der_old,
                                        sycl::buffer<f32> &buf_eps_sq,
                                        sycl::range<1> elem_range,
                                        f32 hdt);

template void integ::leapfrog_corrector(sycl::queue &queue,
                                        sycl::buffer<f64_3> &buf_val,
                                        sycl::buffer<f64_3> &buf_der,
                                        sycl::buffer<f64_3> &buf_der_old,
                                        sycl::buffer<f64> &buf_eps_sq,
                                        sycl::range<1> elem_range,
                                        f64 hdt);

template void integ::leapfrog_corrector(sycl::queue &queue,
                                        sycl::buffer<f64> &buf_val,
                                        sycl::buffer<f64> &buf_der,
                                        sycl::buffer<f64> &buf_der_old,
                                        sycl::buffer<f64> &buf_eps_sq,
                                        sycl::range<1> elem_range,
                                        f64 hdt);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::sycl_position_modulo(sycl::queue &queue,
                                sycl::buffer<T> &buf_xyz,
                                sycl::range<1> elem_range,
                                std::pair<T, T> box) {

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor xyz{buf_xyz, cgh, sycl::read_write};

        T box_min = std::get<0>(box);
        T box_max = std::get<1>(box);
        T delt    = box_max - box_min;

        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            T r = xyz[gid] - box_min;

            r = sycl::fmod(r, delt);
            r += delt;
            r = sycl::fmod(r, delt);
            r += box_min;

            xyz[gid] = r;
        });
    });
}

template void util::sycl_position_modulo(sycl::queue &queue,
                                         sycl::buffer<f32_3> &buf_xyz,
                                         sycl::range<1> elem_range,
                                         std::pair<f32_3, f32_3> box);

template void util::sycl_position_modulo(sycl::queue &queue,
                                         sycl::buffer<f64_3> &buf_xyz,
                                         sycl::range<1> elem_range,
                                         std::pair<f64_3, f64_3> box);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::sycl_position_sheared_modulo(sycl::queue &queue,
                                        sycl::buffer<T> &buf_xyz,
                                        sycl::buffer<T> &buf_vxyz,
                                        sycl::range<1> elem_range,
                                        std::pair<T, T> box,
                                        i32_3 shear_base,
                                        i32_3 shear_dir,
                                        shambase::VecComponent<T> shear_value,
                                        shambase::VecComponent<T> shear_speed) {

    queue.submit([&,shear_base,shear_value,shear_speed](sycl::handler &cgh) {
        sycl::accessor xyz{buf_xyz, cgh, sycl::read_write};
        sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_write};

        T box_min = std::get<0>(box);
        T box_max = std::get<1>(box);
        T delt    = box_max - box_min;

        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            T r = xyz[gid] - box_min;

            T roff = r / delt;
            
            i32 xoff = int(roff.x());
            i32 yoff = int(roff.y());
            i32 zoff = int(roff.z());

            i32 dx = xoff*shear_base.x();
            i32 dy = yoff*shear_base.y();
            i32 dz = zoff*shear_base.z();

            i32 d = dx + dy + dz;


            T shift = {
                (d*shear_dir.x())*shear_value + delt.x()*xoff,
                (d*shear_dir.y())*shear_value + delt.y()*yoff ,
                (d*shear_dir.z())*shear_value + delt.z()*zoff
            };


            T shift_speed = {
                (d*shear_dir.x())*shear_speed,
                (d*shear_dir.y())*shear_speed,
                (d*shear_dir.z())*shear_speed
            };

            vxyz[gid] -= shift_speed;

            r -= shift;

            //r = sycl::fmod(r, delt);
            //r += delt;
            //r = sycl::fmod(r, delt);
            r += box_min;

            xyz[gid] = r;
        });
    });
}

template void util::sycl_position_sheared_modulo(sycl::queue &queue,
                                                 sycl::buffer<f32_3> &buf_xyz,
                                                 sycl::buffer<f32_3> &buf_vxyz,
                                                 sycl::range<1> elem_range,
                                                 std::pair<f32_3, f32_3> box,
                                                 i32_3 shear_base,
                                                 i32_3 shear_dir,
                                                 f32 shear_value,
                                                 f32 shear_speed);

template void util::sycl_position_sheared_modulo(sycl::queue &queue,
                                                 sycl::buffer<f64_3> &buf_xyz,
                                                 sycl::buffer<f64_3> &buf_vxyz,
                                                 sycl::range<1> elem_range,
                                                 std::pair<f64_3, f64_3> box,
                                                 i32_3 shear_base,
                                                 i32_3 shear_dir,
                                                 f64 shear_value,
                                                 f64 shear_speed);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void util::swap_fields(sycl::queue &queue, sycl::buffer<T> &b1, sycl::buffer<T> &b2, u32 cnt) {

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor acc1{b1, cgh, sycl::read_write};
        sycl::accessor acc2{b2, cgh, sycl::read_write};

        cgh.parallel_for(sycl::range<1>{cnt}, [=](sycl::item<1> item) {
            T v1 = acc1[item];
            T v2 = acc2[item];

            acc1[item] = v2;
            acc2[item] = v1;
        });
    });
}

template void
util::swap_fields(sycl::queue &queue, sycl::buffer<f32_3> &b1, sycl::buffer<f32_3> &b2, u32 cnt);

template void
util::swap_fields(sycl::queue &queue, sycl::buffer<f32> &b1, sycl::buffer<f32> &b2, u32 cnt);

template void
util::swap_fields(sycl::queue &queue, sycl::buffer<f64_3> &b1, sycl::buffer<f64_3> &b2, u32 cnt);

template void
util::swap_fields(sycl::queue &queue, sycl::buffer<f64> &b1, sycl::buffer<f64> &b2, u32 cnt);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file modifiers.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <stdexcept>
#include <tuple>
#include <vector>

//%Impl status : Good

namespace generic::setup::modifiers {

    template<class T, class vec>
    inline void set_value_in_box(
        PatchScheduler &sched, T val, std::string name, std::tuple<vec, vec> box) {
        StackEntry stack_loc{};
        sched.patch_data.for_each_patchdata([&](u64 patch_id,
                                                shamrock::patch::PatchDataLayer &pdat) {
            PatchDataField<vec> &xyz
                = pdat.template get_field<vec>(sched.pdl().get_field_idx<vec>("xyz"));

            PatchDataField<T> &f = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(name));

            {
                auto &buf = f.get_buf();
                sycl::host_accessor acc{*buf};

                auto &buf_xyz = xyz.get_buf();
                sycl::host_accessor acc_xyz{*buf_xyz};

                for (u32 i = 0; i < f.size(); i++) {
                    vec r = acc_xyz[i];

                    if (BBAA::is_coord_in_range(r, std::get<0>(box), std::get<1>(box))) {
                        acc[i] = val;
                    }
                }
            }
        });
    }

    template<class T, class vec>
    inline void set_value_in_sphere(
        PatchScheduler &sched,
        T val,
        std::string name,
        vec center,
        shambase::VecComponent<vec> radius) {

        using flt = shambase::VecComponent<vec>;

        StackEntry stack_loc{};
        sched.patch_data.for_each_patchdata([&](u64 patch_id,
                                                shamrock::patch::PatchDataLayer &pdat) {
            PatchDataField<vec> &xyz
                = pdat.template get_field<vec>(sched.pdl().get_field_idx<vec>("xyz"));

            PatchDataField<T> &f = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(name));

            flt r2 = radius * radius;
            {
                auto &buf = f.get_buf();
                sycl::host_accessor acc{*buf};

                auto &buf_xyz = xyz.get_buf();
                sycl::host_accessor acc_xyz{*buf_xyz};

                for (u32 i = 0; i < f.size(); i++) {
                    vec dr = acc_xyz[i] - center;

                    if (sycl::dot(dr, dr) < r2) {
                        acc[i] = val;
                    }
                }
            }
        });
    }

    template<class flt>
    inline void pertub_eigenmode_wave(
        PatchScheduler &sched, std::tuple<flt, flt> ampls, sycl::vec<flt, 3> k, flt phase) {

        using vec = sycl::vec<flt, 3>;

        if (std::get<0>(ampls) != 0) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "density perturbation not implemented");
        }

        sched.patch_data.for_each_patchdata(
            [&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
                PatchDataField<vec> &xyz
                    = pdat.template get_field<vec>(sched.pdl().get_field_idx<vec>("xyz"));
                PatchDataField<vec> &vxyz
                    = pdat.template get_field<vec>(sched.pdl().get_field_idx<vec>("vxyz"));

                flt ampl = std::get<1>(ampls);

                {

                    u32 cnt = pdat.get_obj_cnt();

                    auto &buf_xyz = xyz.get_buf();
                    auto acc_xyz  = buf_xyz.copy_to_stdvec();

                    auto &buf_vxyz = vxyz.get_buf();
                    auto acc_vxyz  = buf_vxyz.copy_to_stdvec();

                    for (u32 i = 0; i < cnt; i++) {
                        vec r       = acc_xyz[i];
                        flt rkphi   = sycl::dot(r, k) + phase;
                        acc_vxyz[i] = ampl * sycl::sin(rkphi);
                    }

                    buf_xyz.copy_from_stdvec(acc_xyz);
                    buf_vxyz.copy_from_stdvec(acc_vxyz);
                }
            });
    }

    template<class T>
    inline T get_sum(PatchScheduler &sched, std::string name) {

        T sum = shambase::VectorProperties<T>::get_zero();

        StackEntry stack_loc{};
        sched.patch_data.for_each_patchdata([&](u64 patch_id,
                                                shamrock::patch::PatchDataLayer &pdat) {
            PatchDataField<T> &xyz = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(name));

            sum += xyz.compute_sum();
        });

        return shamalgs::collective::allreduce_sum(sum);
    }

} // namespace generic::setup::modifiers

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierApplyDiscWarp.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyDiscWarp.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer
shammodels::sph::modules::ModifierApplyDiscWarp<Tvec, SPHKernel>::next_n(u32 nmax) {

    using Config = SolverConfig<Tvec, SPHKernel>;
    Config solver_config;
    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    ////////////////////////// constants //////////////////////////
    constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;
    Tscal Rwarp          = Rwarp_;
    Tscal Hwarp          = Hwarp_;
    Tscal inclination    = inclination_;
    Tscal posangle       = posangle_;

    ////////////////////////// load data //////////////////////////
    auto &pdl                         = sched.pdl();
    sham::DeviceBuffer<Tvec> &buf_xyz = tmp.get_field_buf_ref<Tvec>(pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(pdl.get_field_idx<Tvec>("vxyz"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_xyz  = buf_xyz.get_write_access(depends_list);
    auto acc_vxyz = buf_vxyz.get_write_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, tmp.get_obj_cnt(), "Warp", [=](i32 id_a) {
            Tvec &xyz_a  = acc_xyz[id_a];
            Tvec &vxyz_a = acc_vxyz[id_a];

            Tscal r = sycl::sqrt(sycl::dot(xyz_a, xyz_a));

            Tvec k    = Tvec(-sycl::sin(posangle), sycl::cos(posangle), 0.);
            Tscal psi = 0.;

            // convert to radians (sycl functions take radians)
            Tscal incl_rad = inclination * shambase::constants::pi<Tscal> / 180.;

            Tscal effective_inc;
            if (r < Rwarp - Hwarp) {
                effective_inc = 0.;
            } else if (r < Rwarp + Hwarp && r > Rwarp - Hwarp) {
                effective_inc = sycl::asin(
                    0.5
                    * (1. + sycl::sin(shambase::constants::pi<Tscal> / (2. * Hwarp) * (r - Rwarp)))
                    * sycl::sin(incl_rad));
                psi = shambase::constants::pi<Tscal> * Rwarp / (4. * Hwarp) * sycl::sin(incl_rad)
                      / sycl::sqrt(1. - (0.5 * sycl::pown(sycl::sin(incl_rad), 2)));
                Tscal psimax = sycl::max(psimax, psi);
            } else {
                effective_inc = incl_rad;
            }

            Tvec w  = sycl::cross(k, xyz_a);
            Tvec wv = sycl::cross(k, vxyz_a);
            // Rodrigues' rotation formula
            xyz_a = xyz_a * sycl::cos(effective_inc) + w * sycl::sin(effective_inc)
                    + k * sycl::dot(k, xyz_a) * (1. - sycl::cos(effective_inc));
            vxyz_a = vxyz_a * sycl::cos(effective_inc) + wv * sycl::sin(effective_inc)
                     + k * sycl::dot(k, vxyz_a) * (1. - sycl::cos(effective_inc));
        });
    });

    buf_xyz.complete_event_state(e);
    buf_vxyz.complete_event_state(e);

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M4>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M6>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, M8>;

template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, C2>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, C4>;
template class shammodels::sph::modules::ModifierApplyDiscWarp<f64_3, C6>;

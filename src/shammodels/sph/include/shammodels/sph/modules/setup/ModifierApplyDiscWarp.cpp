
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
 * @file ModifierApplyDiscWarp.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "ModifierApplyDiscWarp.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchData

shammodels::sph::modules::ModifierApplyDiscWarp<Tvec, SPHKernel>::next_n(u32 nmax) {

    using Solver = shammodels::sph::Solver<Tvec, SPHKernel>;
    Solver solver;
    ShamrockCtx &ctx               = context;
    PatchScheduler &sched          = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchData tmp = parent->next_n(nmax);

    ////////////////////////// constants //////////////////////////
    constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;
    Tscal central_mass   = 1.;
    Tscal Rwarp          = 1.;
    Tscal Hwarp          = 1.;
    Tscal theta          = 1.;
    Tscal Gauss          = 1.;
    Tscal posangle       = 1.;
    Tscal incl           = 1;
    Tscal G              = solver.solver_config.get_constant_G();

    if (!is_done()) {
        logger::debug_ln("Warping the disc");
    }

    ////////////////////////// load data //////////////////////////
    PatchDataField<Tvec> &buf_xyz  = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
    PatchDataField<Tvec> &buf_vxyz = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
    PatchDataField<Tvec> &buf_cs = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("soundspeed"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parralel_for(cgh, tmp.get_obj_cnt(), "Warp", [=](i32 id_a) {
            Tvec xyz_a  = buf_xyz[id_a];
            Tvec vxyz_a = buf_vxyz[id_a];
            Tscal cs    = buf_cs[id_a];

            Tscal r = sycl::sqrt(sycl::dot(xyz_a, xyz_a));

            Tscal Omega_Kep = sycl::sqrt(G * central_mass / (r * r * r));
            Tscal H         = cs / Omega_Kep; // sycl::sqrt(2.) * 3. *
            Tscal z         = H * Gauss;

            Tscal vtheta = rot_profile(r);
            Tscal sigma  = sigma_profile(r);

            auto pos = sycl::vec<Tscal, 3>{r * sycl::cos(theta), r * sycl::sin(theta), z};
            auto vel = sycl::vec<Tscal, 3>{
                -vtheta * sycl::sin(theta), vtheta * sycl::cos(theta), 0.}; // vk*etheta;

            Tvec k = Tvec(-std::sin(posangle), std::cos(posangle), 0.);
            Tscal inc;
            Tscal psi = 0.;

            // convert to radians (sycl functions take radians)
            Tscal incl_rad = incl * shambase::constants::pi<Tscal> / 180.;

            Tscal fs  = 1.; // 1. - sycl::sqrt(r_in / r);
            Tscal rho = (sigma * fs) / (sycl::sqrt(_2pi) * H) * sycl::exp(-z * z / (2 * H * H));

            if (r < Rwarp - Hwarp) {
                inc = 0.;
            } else if (r < Rwarp + 3. * Hwarp && r > Rwarp - Hwarp) {
                inc = sycl::asin(
                    0.5
                    * (1. + sycl::sin(shambase::constants::pi<Tscal> / (2. * Hwarp) * (r - Rwarp)))
                    * sycl::sin(incl_rad));
                psi = shambase::constants::pi<Tscal> * Rwarp / (4. * Hwarp) * sycl::sin(incl_rad)
                      / sycl::sqrt(1. - (0.5 * sycl::pow(sycl::sin(incl_rad), 2)));
                Tscal psimax = sycl::max(psimax, psi);
                Tscal x      = pos.x();
                Tscal y      = pos.y();
                Tscal z      = pos.z();
                Tvec kk      = Tvec(0., 0., 1.);
                Tvec w       = sycl::cross(k, pos);
                Tvec wv      = sycl::cross(k, vel);
                // Rodrigues' rotation formula
                pos = pos * sycl::cos(inc) + w * sycl::sin(inc)
                      + k * sycl::dot(k, pos) * (1. - sycl::cos(inc));
                vel = vel * sycl::cos(inc) + wv * sycl::sin(inc)
                      + k * sycl::dot(k, vel) * (1. - sycl::cos(inc));
            } else {
                inc = 0.;
            }
        });
    });

    return tmp;
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierApplyStretchMapping.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 *
 */

#include "shammodels/sph/modules/setup/ModifierApplyStretchMapping.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <cstddef>

// template<class Tvec>

// template<class Tvec, template<class> class SPHKernel>
// Tvec shammodels::sph::modules::ModifierApplyStretchMapping<Tvec, SPHKernel>::
//     ModifierApplyStretchMapping<Tvec, SPHKernel>::testvec(
//         Tvec x, std::vector<std::function<Tscal(Tscal)>> rhoprofiles) {
//     return x * 2.;
// }

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer shammodels::sph::modules::ModifierApplyStretchMapping<
    Tvec,
    SPHKernel>::ModifierApplyStretchMapping<Tvec, SPHKernel>::next_n(u32 nmax) {

    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    // No objects to offset
    if (tmp.get_obj_cnt() == 0) {
        return tmp;
    }

    // // auto &rhoprofiles       = this->rhoprofiles;
    // auto &axes              = this->axes;
    // auto &rhodSs            = this->rhodSs;
    // auto &a_from_poss       = this->a_from_poss;
    // auto &a_to_poss         = this->a_to_poss;
    // auto &integral_profiles = this->integral_profiles;
    // auto &ximins            = this->ximins;
    // auto &ximaxs            = this->ximaxs;
    // auto &center            = this->center;
    // auto &steps             = this->steps;

    // auto &mpart = this->mpart;
    // auto &hfact = this->hfact;

    // auto &stretchpart     = this->stretchpart;
    // auto &h_rho_stretched = this->h_rho_stretched;
    // auto &test            = this->test;
    // // auto testvec          = this->testvec;
    // // auto testvec = [](Tvec x) -> Tvec {
    // //     return x * 2.;
    // // };

    // auto &test2 = this->test2;

    // Tvec xyz_a_dbebug = stretchpart(
    //     {1., 1., 1.},
    //     rhoprofiles,
    //     rhodSs,
    //     a_from_poss,
    //     a_to_poss,
    //     integral_profiles,
    //     ximins,
    //     ximaxs,
    //     center,
    //     steps);
    // Tscal hpart_debug
    //     = h_rho_stretched({1., 1., 1.}, rhoprofiles, a_from_poss, integral_profiles, mpart,
    //     hfact);
    // shamlog_debug_ln("kernel_call", xyz_a_dbebug, hpart_debug);
    // // shamlog_debug_ln("kernel_call", test(2), testvec({1, 2, 1}));

    // sham::DeviceBuffer<Tvec> &buf_xyz
    //     = tmp.get_field_buf_ref<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));
    // sham::DeviceBuffer<Tscal> &buf_hpart
    //     = tmp.get_field_buf_ref<Tscal>(sched.pdl().get_field_idx<Tscal>("hpart"));

    // auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    // sham::kernel_call(
    //     q,
    //     sham::MultiRef{},
    //     sham::MultiRef{buf_xyz, buf_hpart},
    //     tmp.get_obj_cnt(),
    //     [&](u32 i, Tvec *__restrict__ xyz_a, Tscal *__restrict__ hpart) {
    //         // xyz_a[i] = xyz_a[i] + 1;
    //         // hpart[i] = hpart[i] + 1;
    //         xyz_a[i] = stretchpart(
    //             xyz_a[i],
    //             rhoprofiles,
    //             rhodSs,
    //             a_from_poss,
    //             a_to_poss,
    //             integral_profiles,
    //             ximins,
    //             ximaxs,
    //             center,
    //             steps);
    //         hpart[i] = h_rho_stretched(
    //             xyz_a[i], rhoprofiles, a_from_poss, integral_profiles, mpart, hfact);
    //     });
    // auto rhoprofiltest = this->rhoprofiletest;

    // auto lrhoprofiltest = [rhoprofiltest = this->rhoprofiletest](Tscal x) -> Tscal {
    //     return rhoprofiltest(x);
    // };

    // // using testvec = testvec<Tvec>;
    // // using testvec_fn = ::testvec<Tvec>;
    // sham::kernel_call(
    //     q,
    //     sham::MultiRef{},
    //     sham::MultiRef{buf_xyz},
    //     tmp.get_obj_cnt(),
    //     [this, lrhoprofiltest](u32 i, Tvec *__restrict__ xyz_a) {
    //         // xyz_a[i] = testvec(xyz_a[i], rhoprofiles);
    //         auto x   = rhoprofiletest(0);
    //         xyz_a[i] = testvec(xyz_a[i]);
    //     });

    auto &pdl                         = sched.pdl();
    sham::DeviceBuffer<Tvec> &buf_xyz = tmp.get_field_buf_ref<Tvec>(pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tscal> &buf_hpart
        = tmp.get_field_buf_ref<Tscal>(pdl.get_field_idx<Tscal>("hpart"));

    auto acc_xyz   = buf_xyz.copy_to_stdvec();
    auto acc_hpart = buf_hpart.copy_to_stdvec();

    Tscal npart = 0;

    for (i32 id_a = 0; id_a < tmp.get_obj_cnt(); ++id_a) {
        Tvec &xyz_a    = acc_xyz[id_a];
        Tscal &hpart_a = acc_hpart[id_a];

        xyz_a = stretchpart(
            xyz_a,
            rhoprofiles,
            rhodSs,
            a_from_poss,
            a_to_poss,
            integral_profiles,
            ximins,
            ximaxs,
            center,
            steps);
        hpart_a = h_rho_stretched(xyz_a, rhoprofiles, a_from_poss, integral_profiles, mpart, hfact);

        npart += 1;
    };

    buf_xyz.copy_from_stdvec(acc_xyz);
    buf_hpart.copy_from_stdvec(acc_hpart);

    shamlog_debug_ln("ModifierApplyStretchMapping", npart, "particles have been stretched");

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::kernel_call(
        q,
        sham::MultiRef{},
        sham::MultiRef{buf_hpart},
        tmp.get_obj_cnt(),
        [npart](u32 i, Tscal *__restrict hpart) {
            hpart[i] = hpart[i] / sycl::rootn(npart, 3);
        });

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M4>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M6>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M8>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C2>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C4>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C6>;

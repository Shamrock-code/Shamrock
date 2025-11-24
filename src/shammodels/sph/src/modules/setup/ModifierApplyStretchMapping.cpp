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
#include "shambackends/kernel_call.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer shammodels::sph::modules::
    ModifierApplyStretchMapping<Tvec, SPHKernel>::next_n(u32 nmax) {

    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    // No objects to offset
    if (tmp.get_obj_cnt() == 0) {
        return tmp;
    }

    sham::DeviceBuffer<Tvec> &buf_xyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tscal> &buf_hpart
        = tmp.get_field_buf_ref<Tscal>(sched.pdl().get_field_idx<Tscal>("hpart"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    std::vector<std::function<Tscal(Tscal)>> &rhoprofiles    = this->rhoprofiles;
    std::vector<std::string> &axes                           = this->axes;
    std::vector<std::function<Tscal(Tscal)>> &rhodSs         = this->rhodSs;
    std::vector<std::function<Tscal(Tvec)>> &a_from_poss     = this->a_from_poss;
    std::vector<std::function<Tvec(Tscal, Tvec)>> &a_to_poss = this->a_to_poss;
    std::vector<Tscal> &integral_profiles                    = this->integral_profiles;
    std::vector<Tscal> &ximins                               = this->ximins;
    std::vector<Tscal> &ximaxs                               = this->ximaxs;
    Tvec &center                                             = this->center;
    std::vector<Tscal> &steps                                = this->steps;

    Tscal mpart = this->mpart;
    Tscal hfact = this->hfact;

    auto &stretchpart     = this->stretchpart;
    auto &h_rho_stretched = this->h_rho_stretched;

    Tvec xyz_a_dbebug = stretchpart(
        {1., 1., 1.},
        rhoprofiles,
        rhodSs,
        a_from_poss,
        a_to_poss,
        integral_profiles,
        ximins,
        ximaxs,
        center,
        steps);
    Tscal hpart_debug
        = h_rho_stretched({1., 1., 1.}, rhoprofiles, a_from_poss, integral_profiles, mpart, hfact);
    shamlog_debug_ln("kernel_call", xyz_a_dbebug, hpart_debug);

    sham::kernel_call(
        q,
        sham::MultiRef{},
        sham::MultiRef{buf_xyz, buf_hpart},
        tmp.get_obj_cnt(),
        [=](u32 i, Tvec *__restrict__ xyz_a, Tscal *__restrict__ hpart) {
            // xyz_a[i] = xyz_a[i] + 1;
            // hpart[i] = hpart[i] + 1;
            xyz_a[i] = stretchpart(
                xyz_a[i],
                rhoprofiles,
                rhodSs,
                a_from_poss,
                a_to_poss,
                integral_profiles,
                ximins,
                ximaxs,
                center,
                steps);
            hpart[i] = h_rho_stretched(
                xyz_a[i], rhoprofiles, a_from_poss, integral_profiles, mpart, hfact);
        });

    // sham::EventList depends_list;
    // auto acc_xyz   = buf_xyz.get_write_access(depends_list);
    // auto acc_hpart = buf_hpart.get_write_access(depends_list);

    // auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
    //     shambase::parallel_for(cgh, tmp.get_obj_cnt(), "Warp", [&](i32 id_a) {
    //         acc_xyz[id_a] = stretchpart(
    //             acc_xyz[id_a],
    //             rhoprofiles,
    //             rhodSs,
    //             a_from_poss,
    //             a_to_poss,
    //             integral_profiles,
    //             ximins,
    //             ximaxs,
    //             center,
    //             steps);
    //         ;
    //         acc_hpart[id_a] = h_rho_stretched(
    //             acc_xyz[id_a], rhoprofiles, a_from_poss, integral_profiles, mpart, hfact);
    //         ;
    //     });
    // });

    // buf_xyz.complete_event_state(e);
    // buf_hpart.complete_event_state(e);

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M4>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M6>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, M8>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C2>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C4>;
template class shammodels::sph::modules::ModifierApplyStretchMapping<f64_3, C6>;

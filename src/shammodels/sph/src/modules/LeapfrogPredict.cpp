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
 * @file LeapfrogPredict.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/LeapfrogPredict.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void LeapfrogPredict<Tvec, SPHKernel>::do_predictor(Tscal dt) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;
        PatchDataLayout &pdl = scheduler().pdl;
        const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");
        const u32 iuint      = pdl.get_field_idx<Tscal>("uint");
        const u32 iduint     = pdl.get_field_idx<Tscal>("duint");

        bool has_B_field       = solver_config.has_field_B_on_rho();
        bool has_psi_field     = solver_config.has_field_psi_on_ch();
        bool has_epsilon_field = solver_config.dust_config.has_epsilon_field();
        bool has_deltav_field  = solver_config.dust_config.has_deltav_field();

        const u32 iB_on_rho   = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : 0;
        const u32 idB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
        const u32 ipsi_on_ch  = (has_psi_field) ? pdl.get_field_idx<Tscal>("psi/ch") : 0;
        const u32 idpsi_on_ch = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;

        const u32 iepsilon   = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("epsilon") : 0;
        const u32 idtepsilon = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("dtepsilon") : 0;
        const u32 ideltav    = (has_deltav_field) ? pdl.get_field_idx<Tvec>("deltav") : 0;
        const u32 idtdeltav  = (has_deltav_field) ? pdl.get_field_idx<Tvec>("dtdeltav") : 0;

        shamrock::SchedulerUtility utility(scheduler());

        // forward euler step f dt/2
        logger::debug_ln("sph::BasicGas", "forward euler step f dt/2");
        utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

        if (has_B_field) {
            utility.fields_forward_euler<Tvec>(iB_on_rho, idB_on_rho, dt / 2);
        }
        if (has_psi_field) {
            utility.fields_forward_euler<Tscal>(ipsi_on_ch, idpsi_on_ch, dt / 2);
        }

        // forward euler step positions dt
        logger::debug_ln("sph::BasicGas", "forward euler step positions dt");
        utility.fields_forward_euler<Tvec>(ixyz, ivxyz, dt);

        // forward euler step f dt/2
        logger::debug_ln("sph::BasicGas", "forward euler step f dt/2");
        utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
        utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

        if (has_B_field) {
            utility.fields_forward_euler<Tvec>(iB_on_rho, idB_on_rho, dt / 2);
        }
        if (has_psi_field) {
            utility.fields_forward_euler<Tscal>(ipsi_on_ch, idpsi_on_ch, dt / 2);
        }
        if (has_epsilon_field) {
            utility.fields_forward_euler<Tscal>(iepsilon, idtepsilon, dt / 2);
        }
        if (has_deltav_field) {
            utility.fields_forward_euler<Tvec>(ideltav, idtdeltav, dt / 2);
        }
    };

} // namespace shammodels::sph::modules

using namespace shammath;
template class shammodels::sph::modules::LeapfrogPredict<f64_3, M4>;
template class shammodels::sph::modules::LeapfrogPredict<f64_3, M6>;
template class shammodels::sph::modules::LeapfrogPredict<f64_3, M8>;

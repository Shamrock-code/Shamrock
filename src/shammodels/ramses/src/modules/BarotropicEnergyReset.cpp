// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BarotropicEnergyReset.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/BarotropicEnergyReset.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/riemann_common.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::BarotropicEnergyReset<Tvec, TgridVec>::reset_energy() {

    StackEntry stack_loc{};

    using namespace shamrock::patch;

    using ResetBarotropic      = typename EOSConfig<Tvec>::ResetBarotropic;
    const ResetBarotropic *cfg = std::get_if<ResetBarotropic>(&solver_config.eos_config.config);
    if (cfg == nullptr) {
        return;
    }

    const shammath::FluidStateBarotropic<Tvec> barotrope = cfg->get_barotrope();
    const Tscal gamma                                    = cfg->gamma;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 irho     = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel  = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln("[AMR Flux]", "barotropic energy reset patch", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            sham::kernel_call(
                q,
                sham::MultiRef{buf_rho, buf_rhov},
                sham::MultiRef{buf_rhoe},
                cell_count,
                [barotrope, gamma](
                    u32 id_a,
                    const Tscal *__restrict rho,
                    const Tvec *__restrict rhov,
                    Tscal *__restrict rhoe) {
                    // the barotropic cons_to_prim ignores rhoe and sets press = P_baro(rho)
                    auto prim = barotrope.cons_to_prim(
                        shammath::ConsState<Tvec>{rho[id_a], rhoe[id_a], rhov[id_a]});

                    rhoe[id_a] = shammath::prim_to_cons(prim, gamma).rhoe;
                });
        });
}

template class shammodels::basegodunov::modules::BarotropicEnergyReset<f64_3, i64_3>;

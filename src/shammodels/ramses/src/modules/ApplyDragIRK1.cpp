// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyDragIRK1.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ApplyDragIRK1.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyDragIRK1<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe_next.check_sizes(edges.sizes.indexes);
        edges.spans_rho_d_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_d_next.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);
        edges.spans_rho_d.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_d.check_sizes(edges.sizes.indexes);

        Tscal dt             = edges.dt.value;
        u32 ndust            = this->ndust;
        u32 friction_control = (enable_frictional_heating == false) ? 1 : 0;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::DeviceBuffer<Tscal> alphas_buf(ndust, dev_sched);
        alphas_buf.copy_from_stdvec(alphas);

        edges.sizes.indexes.for_each([&](u64 id_patch, u32 block_count) {
            u32 cell_count = block_count * block_size;

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{
                    edges.spans_rho_next.get_spans().get(id_patch),
                    edges.spans_rhov_next.get_spans().get(id_patch),
                    edges.spans_rhoe_next.get_spans().get(id_patch),
                    edges.spans_rho_d_next.get_spans().get(id_patch),
                    edges.spans_rhov_d_next.get_spans().get(id_patch),
                    alphas_buf},
                sham::MultiRef{
                    edges.spans_rho.get_spans().get(id_patch),
                    edges.spans_rhov.get_spans().get(id_patch),
                    edges.spans_rhoe.get_spans().get(id_patch),
                    edges.spans_rho_d.get_spans().get(id_patch),
                    edges.spans_rhov_d.get_spans().get(id_patch)},
                cell_count,
                [dt, ndust, friction_control](
                    u32 id_a,
                    const Tscal *__restrict rho_next,
                    const Tvec *__restrict rhov_next,
                    const Tscal *__restrict rhoe_next,
                    const Tscal *__restrict rho_d_next,
                    const Tvec *__restrict rhov_d_next,
                    const Tscal *__restrict alphas,
                    Tscal *__restrict rho,
                    Tvec *__restrict rhov,
                    Tscal *__restrict rhoe,
                    Tscal *__restrict rho_d,
                    Tvec *__restrict rhov_d) {
                    Tvec tmp_mom_1 = rhov_next[id_a];
                    Tscal tmp_rho  = rho[id_a];

                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alphas[i] * dt);
                        const Tscal dt_alphas     = dt * alphas[i];

                        tmp_mom_1
                            = tmp_mom_1 + dt_alphas * inv_dt_alphas * rhov_d_next[id_a * ndust + i];
                        tmp_rho = tmp_rho + dt_alphas * inv_dt_alphas * rho_d[id_a * ndust + i];
                    }

                    Tscal tmp_inv_rho = 1.0 / tmp_rho;
                    Tvec tmp_vel      = tmp_inv_rho * tmp_mom_1;
                    Tscal Eg          = 0.0;

                    Tscal inv_rho_g = 1.0 / rho_next[id_a];
                    Tvec vg_bf      = inv_rho_g * rhov_next[id_a];
                    Tvec vg_af      = inv_rho_g * rho[id_a] * tmp_vel;

                    Tscal work_drag
                        = 0.5
                          * ((rho[id_a] * tmp_vel[0] - rhov_next[id_a][0]) * (vg_bf[0] + vg_af[0])
                             + (rho[id_a] * tmp_vel[1] - rhov_next[id_a][1]) * (vg_bf[1] + vg_af[1])
                             + (rho[id_a] * tmp_vel[2] - rhov_next[id_a][2])
                                   * (vg_bf[2] + vg_af[2]));
                    Tscal dissipation = 0.0;
                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alphas[i] * dt);
                        const Tscal dt_alphas     = dt * alphas[i];
                        Tscal inv_rho_d           = 1.0 / rho_d_next[id_a * ndust + i];
                        Tvec vd_bf                = inv_rho_d * rhov_d_next[id_a * ndust + i];
                        Tvec vd_af = inv_rho_d * inv_dt_alphas
                                     * (rhov_d_next[id_a * ndust + i]
                                        + dt_alphas * rho_d[id_a * ndust + i] * tmp_vel);
                        dissipation += 0.5 * dt_alphas * inv_dt_alphas
                                       * ((rho_d[id_a * ndust + i] * tmp_vel[0]
                                           - rhov_d_next[id_a * ndust + i][0])
                                              * (vd_af[0] + vd_bf[0])
                                          + (rho_d[id_a * ndust + i] * tmp_vel[1]
                                             - rhov_d_next[id_a * ndust + i][1])
                                                * (vd_af[1] + vd_bf[1])
                                          + (rho_d[id_a * ndust + i] * tmp_vel[2]
                                             - rhov_d_next[id_a * ndust + i][2])
                                                * (vd_af[2] + vd_bf[2]));
                    }

                    Eg += rhoe_next[id_a] + (1 - friction_control) * work_drag
                          - friction_control * dissipation;
                    rhov[id_a] = tmp_vel * rho[id_a];
                    rhoe[id_a] = Eg;
                    rho[id_a]  = rho_next[id_a];
                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alphas[i] * dt);
                        const Tscal dt_alphas     = dt * alphas[i];
                        rhov_d[id_a * ndust + i]
                            = inv_dt_alphas
                              * (rhov_d_next[id_a * ndust + i]
                                 + dt_alphas * rho_d[id_a * ndust + i] * tmp_vel);
                        rho_d[id_a * ndust + i] = rho_d_next[id_a * ndust + i];
                    }
                });
        });
    }

    template<class Tvec>
    std::string ApplyDragIRK1<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Gas-dust drag, first order implicit Runge-Kutta (IRK1)

            Update ${spans_rho}$, ${spans_rhov}$, ${spans_rhoe}$, ${spans_rho_d}$, ${spans_rhov_d}$ from
            the state without drag ${spans_rho_next}$, ${spans_rhov_next}$, ${spans_rhoe_next}$,
            ${spans_rho_d_next}$, ${spans_rhov_d_next}$ over a time step ${dt}$.
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyDragIRK1<f64_3>;

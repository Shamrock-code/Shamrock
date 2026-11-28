// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyExtForceDustTrapezoidal.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the external forces accelerations at t^n and t^{n+1} to the dust
 * conservative state (trapezoidal rule)
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyExtForceDustTrapezoidal.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyExtForceDustTrapezoidal<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext_n.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext_np1.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_dust.check_sizes(edges.sizes.indexes);

        Tscal dt       = edges.dt.value;
        Tscal dt_over2 = dt / 2;

        shambase::DistributedData<u32> dust_cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size * ndust;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_rho_dust.get_spans(),
                edges.spans_dt_rho_dust.get_spans(),
                edges.spans_a_ext_n.get_spans(),
                edges.spans_a_ext_np1.get_spans()},
            sham::DDMultiRef{edges.spans_rhov_dust.get_spans()},
            dust_cell_counts,
            [dt, dt_over2, ndust = ndust](
                u32 i,
                const Tscal *__restrict rho_dust,
                const Tscal *__restrict dt_rho_dust,
                const Tvec *__restrict a_ext_n,
                const Tvec *__restrict a_ext_np1,
                Tvec *__restrict rhov_dust) {
                // dust fields are laid out as [cell * ndust + idust]
                u32 cell_id = i / ndust;

                Tscal rho_n = rho_dust[i];

                // dust density at t^{n+1}
                Tscal rho_np1 = rho_n + dt * dt_rho_dust[i];

                rhov_dust[i]
                    += dt_over2 * (rho_n * a_ext_n[cell_id] + rho_np1 * a_ext_np1[cell_id]);
            });
    }

    template<class Tvec>
    std::string ApplyExtForceDustTrapezoidal<Tvec>::_impl_get_tex() const {
        auto dt          = get_ro_edge_base(0).get_tex_symbol();
        auto rho_dust    = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rho_dust = get_ro_edge_base(3).get_tex_symbol();
        auto a_ext_n     = get_ro_edge_base(4).get_tex_symbol();
        auto a_ext_np1   = get_ro_edge_base(5).get_tex_symbol();
        auto rhov_dust   = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Apply external forces to the dust species (trapezoidal rule)

            \begin{align}
            \rho^{n+1}_{d,i} &= {rho_dust}_i + {dt} \, {dt_rho_dust}_i \
            {rhov_dust}_i &= {rhov_dust}_i + \frac{{dt}}{2} \left( {rho_dust}_i {a_ext_n}_i
                + \rho^{n+1}_{d,i} {a_ext_np1}_i \right)
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{rho_dust}", rho_dust);
        shambase::replace_all(tex, "{dt_rho_dust}", dt_rho_dust);
        shambase::replace_all(tex, "{a_ext_np1}", a_ext_np1);
        shambase::replace_all(tex, "{a_ext_n}", a_ext_n);
        shambase::replace_all(tex, "{rhov_dust}", rhov_dust);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyExtForceDustTrapezoidal<f64_3>;

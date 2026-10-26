// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyExtForceDust.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyExtForceDust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyExtForceDust<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_dust.check_sizes(edges.sizes.indexes);

        Tscal dt_over2 = edges.dt_over2.value;
        Tscal dt       = 2 * dt_over2;

        shambase::DistributedData<u32> dust_cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size * ndust;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_rho_dust.get_spans(),
                edges.spans_dt_rho_dust.get_spans(),
                edges.spans_a_ext.get_spans()},
            sham::DDMultiRef{edges.spans_rhov_dust.get_spans()},
            dust_cell_counts,
            [dt, dt_over2, ndust = ndust](
                u32 i,
                const Tscal *__restrict rho_dust,
                const Tscal *__restrict dt_rho_dust,
                const Tvec *__restrict a_ext,
                Tvec *__restrict rhov_dust) {
                // dust fields are laid out as [cell * ndust + idust]
                u32 cell_id = i / ndust;

                // time centered dust density at t^{n+1/2}
                Tscal rho_half = rho_dust[i] + dt_over2 * dt_rho_dust[i];

                rhov_dust[i] += dt * rho_half * a_ext[cell_id];
            });
    }

    template<class Tvec>
    std::string ApplyExtForceDust<Tvec>::_impl_get_tex() const {
        auto dt_over2    = get_ro_edge_base(0).get_tex_symbol();
        auto rho_dust    = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rho_dust = get_ro_edge_base(3).get_tex_symbol();
        auto a_ext       = get_ro_edge_base(4).get_tex_symbol();
        auto rhov_dust   = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Apply external forces to the dust species

            \begin{align}
            \rho^{n+1/2}_{d,i} &= {rho_dust}_i + {dt_over2} \, {dt_rho_dust}_i \
            {rhov_dust}_i &= {rhov_dust}_i + 2 {dt_over2} \, \rho^{n+1/2}_{d,i} {a_ext}_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt_over2}", dt_over2);
        shambase::replace_all(tex, "{rho_dust}", rho_dust);
        shambase::replace_all(tex, "{dt_rho_dust}", dt_rho_dust);
        shambase::replace_all(tex, "{a_ext}", a_ext);
        shambase::replace_all(tex, "{rhov_dust}", rhov_dust);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyExtForceDust<f64_3>;

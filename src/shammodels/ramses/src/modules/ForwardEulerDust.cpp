// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ForwardEulerDust.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ForwardEulerDust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ForwardEulerDust<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_dt_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov_dust.check_sizes(edges.sizes.indexes);
        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_dust.check_sizes(edges.sizes.indexes);

        Tscal dt = edges.dt.value;

        shambase::DistributedData<u32> dust_cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size * ndust;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_dt_rho_dust.get_spans(), edges.spans_dt_rhov_dust.get_spans()},
            sham::DDMultiRef{edges.spans_rho_dust.get_spans(), edges.spans_rhov_dust.get_spans()},
            dust_cell_counts,
            [dt](
                u32 i,
                const Tscal *__restrict dt_rho_dust,
                const Tvec *__restrict dt_rhov_dust,
                Tscal *__restrict rho_dust,
                Tvec *__restrict rhov_dust) {
                rho_dust[i] += dt * dt_rho_dust[i];
                rhov_dust[i] += dt * dt_rhov_dust[i];
            });
    }

    template<class Tvec>
    std::string ForwardEulerDust<Tvec>::_impl_get_tex() const {
        auto dt           = get_ro_edge_base(0).get_tex_symbol();
        auto dt_rho_dust  = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rhov_dust = get_ro_edge_base(3).get_tex_symbol();
        auto rho_dust     = get_rw_edge_base(0).get_tex_symbol();
        auto rhov_dust    = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Forward Euler integration of the dust species

            \begin{align}
            {rho_dust}_i &= {rho_dust}_i + {dt} \, {dt_rho_dust}_i \
            {rhov_dust}_i &= {rhov_dust}_i + {dt} \, {dt_rhov_dust}_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{dt_rho_dust}", dt_rho_dust);
        shambase::replace_all(tex, "{dt_rhov_dust}", dt_rhov_dust);
        shambase::replace_all(tex, "{rho_dust}", rho_dust);
        shambase::replace_all(tex, "{rhov_dust}", rhov_dust);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ForwardEulerDust<f64_3>;

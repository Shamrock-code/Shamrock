// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeNextRho.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Out of place version of the NodeNextRho of the `new_amr_plus_grav` branch of
 * Léodasce Sewanou.
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/ramses/modules/ComputeNextRho.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ComputeNextRho<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes_with_ghost.indexes);
        edges.spans_dt_rho.check_sizes(edges.sizes.indexes);
        edges.rho_next.ensure_sizes(edges.sizes_with_ghost.indexes);

        Tscal dt = edges.dt.value;

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes_with_ghost.indexes.for_each([&](u64 id, u32 block_count_with_ghost) {
            auto &rho_span      = edges.spans_rho.get_spans().get(id);
            auto &dt_rho_span   = edges.spans_dt_rho.get_spans().get(id);
            auto &rho_next_span = edges.rho_next.get_spans().get(id);

            u32 cell_count            = edges.sizes.indexes.get(id) * block_size;
            u32 cell_count_with_ghost = block_count_with_ghost * block_size;

            sham::kernel_call(
                q,
                sham::MultiRef{rho_span, dt_rho_span},
                sham::MultiRef{rho_next_span},
                cell_count_with_ghost,
                [dt, cell_count](
                    u32 i,
                    const Tscal *__restrict rho,
                    const Tscal *__restrict dt_rho,
                    Tscal *__restrict rho_next) {
                    rho_next[i] = (i < cell_count) ? rho[i] + dt * dt_rho[i] : rho[i];
                });
        });
    }

    template<class Tvec>
    std::string ComputeNextRho<Tvec>::_impl_get_tex() const {
        auto dt       = get_ro_edge_base(0).get_tex_symbol();
        auto rho      = get_ro_edge_base(3).get_tex_symbol();
        auto dt_rho   = get_ro_edge_base(4).get_tex_symbol();
        auto rho_next = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Density at the end of the time step

            \begin{align}
            {rho_next}_i &= {rho}_i + {dt} {dt_rho}_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{rho_next}", rho_next);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt}", dt);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ComputeNextRho<f64_3>;

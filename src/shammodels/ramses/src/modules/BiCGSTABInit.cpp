// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BiCGSTABInit.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Initialisation step of the BiCGSTAB Poisson solver.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/BiCGSTABInit.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void BiCGSTABInit<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res_bis.check_sizes(edges.sizes.indexes);
        edges.spans_phi_p.check_sizes(edges.sizes.indexes);
        edges.spans_rhs.check_sizes(edges.sizes.indexes);

        auto &q        = shamsys::instance::get_compute_scheduler().get_queue();
        Tscal mean_rho = edges.mean_rho.value;

        edges.cell_neigh_graph.graph.for_each(
            [&](u64 id, const OrientedAMRGraph<Tvec, TgridVec> &oriented_cell_graph) {
                auto &cell_sizes_span  = edges.spans_block_cell_sizes.get_spans().get(id);
                auto &phi_span         = edges.spans_phi.get_spans().get(id);
                auto &rho_span         = edges.spans_rho.get_spans().get(id);
                auto &phi_res_span     = edges.spans_phi_res.get_spans().get(id);
                auto &phi_res_bis_span = edges.spans_phi_res_bis.get_spans().get(id);
                auto &phi_p_span       = edges.spans_phi_p.get_spans().get(id);
                auto &rhs_span         = edges.spans_rhs.get_spans().get(id);

                auto get_graph = [&](Direction dir) -> const AMRGraph & {
                    return shambase::get_check_ref(oriented_cell_graph.graph_links[dir]);
                };

                const AMRGraph &graph_neigh_xp = get_graph(Direction::xp);
                const AMRGraph &graph_neigh_xm = get_graph(Direction::xm);
                const AMRGraph &graph_neigh_yp = get_graph(Direction::yp);
                const AMRGraph &graph_neigh_ym = get_graph(Direction::ym);
                const AMRGraph &graph_neigh_zp = get_graph(Direction::zp);
                const AMRGraph &graph_neigh_zm = get_graph(Direction::zm);

                u32 cell_count = edges.sizes.indexes.get(id) * block_size;

                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        cell_sizes_span,
                        phi_span,
                        rho_span,
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm},
                    sham::MultiRef{phi_res_span, phi_res_bis_span, phi_p_span, rhs_span},
                    cell_count,
                    [block_size = block_size, fourPiG = fourPiG, mean_rho](
                        u32 cell_global_id,
                        const Tscal *__restrict cell_sizes,
                        const Tscal *__restrict phi,
                        const Tscal *__restrict rho,
                        const AMRGraph::ro_access graph_iter_xp,
                        const AMRGraph::ro_access graph_iter_xm,
                        const AMRGraph::ro_access graph_iter_yp,
                        const AMRGraph::ro_access graph_iter_ym,
                        const AMRGraph::ro_access graph_iter_zp,
                        const AMRGraph::ro_access graph_iter_zm,
                        Tscal *__restrict phi_res,
                        Tscal *__restrict phi_res_bis,
                        Tscal *__restrict phi_p,
                        Tscal *__restrict rhs) {
                        Tscal Aphi = laplacian_7pt<Tscal, Tvec>(
                            cell_sizes,
                            block_size,
                            cell_global_id,
                            graph_iter_xp,
                            graph_iter_xm,
                            graph_iter_yp,
                            graph_iter_ym,
                            graph_iter_zp,
                            graph_iter_zm,
                            [=](u32 id) {
                                return phi[id];
                            });

                        // laplacian_7pt is volume integrated, hence the dV factor
                        Tscal delta_cell = cell_sizes[cell_global_id / block_size];
                        Tscal dV         = delta_cell * delta_cell * delta_cell;
                        Tscal b_rhs      = -fourPiG * (rho[cell_global_id] - mean_rho) * dV;

                        phi_res[cell_global_id]     = b_rhs - Aphi;
                        phi_res_bis[cell_global_id] = 0.5 * (b_rhs - Aphi);
                        phi_p[cell_global_id]       = b_rhs - Aphi;
                        rhs[cell_global_id]         = b_rhs;
                    });
            });
    }

    template<class Tvec, class TgridVec>
    std::string BiCGSTABInit<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Initialisation step of BiCGSTAB
            \begin{align}
            {spans_phi_res}_i &= -4 \pi G ({spans_rho}_i - {mean_rho}) \Delta x_i^3 - (A {spans_phi})_i \\
            {spans_phi_res_bis}_i &= {spans_phi_res}_i / 2 \\
            {spans_phi_p}_i &= {spans_phi_res}_i \\
            {spans_rhs}_i &= -4 \pi G ({spans_rho}_i - {mean_rho}) \Delta x_i^3
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::BiCGSTABInit<f64_3, i64_3>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeJacobiPreconditioner.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Diagonal preconditioner of the PCG Poisson solver.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shammodels/ramses/modules/NodeJacobiPreconditioner.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeJacobiPreconditioner<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.cell_neigh_graph.graph.for_each(
            [&](u64 id, const OrientedAMRGraph<Tvec, TgridVec> &oriented_cell_graph) {
                auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                auto &x_span          = edges.spans_x.get_spans().get(id);
                auto &y_span          = edges.spans_y.get_spans().get(id);

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
                        x_span,
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm},
                    sham::MultiRef{y_span},
                    cell_count,
                    [block_size = block_size](
                        u32 cell_global_id,
                        const Tscal *__restrict cell_sizes,
                        const Tscal *__restrict x,
                        const AMRGraph::ro_access graph_iter_xp,
                        const AMRGraph::ro_access graph_iter_xm,
                        const AMRGraph::ro_access graph_iter_yp,
                        const AMRGraph::ro_access graph_iter_ym,
                        const AMRGraph::ro_access graph_iter_zp,
                        const AMRGraph::ro_access graph_iter_zm,
                        Tscal *__restrict y) {
                        Tscal weight = jacobi_weight_7pt<Tscal, Tvec>(
                            cell_sizes,
                            block_size,
                            cell_global_id,
                            graph_iter_xp,
                            graph_iter_xm,
                            graph_iter_yp,
                            graph_iter_ym,
                            graph_iter_zp,
                            graph_iter_zm);
                        y[cell_global_id] = x[cell_global_id] / weight;
                    });
            });
    }

    template<class Tvec, class TgridVec>
    std::string NodeJacobiPreconditioner<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Jacobi preconditioner
            \begin{equation}
            {spans_y}_i = {spans_x}_i / A_{ii}
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeJacobiPreconditioner<f64_3, i64_3>;

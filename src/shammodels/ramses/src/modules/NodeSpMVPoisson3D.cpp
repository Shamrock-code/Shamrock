// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSpMVPoisson3D.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sparse matrix-vector product \f$ A p \f$, \f$ A \f$ being the 3D Poisson matrix.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/memory.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeSpMVPoisson3D<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_in.check_sizes(edges.sizes.indexes);
        edges.spans_out.ensure_sizes(edges.sizes.indexes);

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.cell_neigh_graph.graph.for_each(
            [&](u64 id, const OrientedAMRGraph<Tvec, TgridVec> &oriented_cell_graph) {
                auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                auto &in_span         = edges.spans_in.get_spans().get(id);
                auto &out_span        = edges.spans_out.get_spans().get(id);

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
                        in_span,
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm},
                    sham::MultiRef{out_span},
                    cell_count,
                    [block_size = block_size](
                        u32 cell_global_id,
                        const Tscal *__restrict cell_sizes,
                        const Tscal *__restrict in,
                        const AMRGraph::ro_access graph_iter_xp,
                        const AMRGraph::ro_access graph_iter_xm,
                        const AMRGraph::ro_access graph_iter_yp,
                        const AMRGraph::ro_access graph_iter_ym,
                        const AMRGraph::ro_access graph_iter_zp,
                        const AMRGraph::ro_access graph_iter_zm,
                        Tscal *__restrict out) {
                        out[cell_global_id] = laplacian_7pt<Tscal, Tvec>(
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
                                return sycl::isnan(in[id]) ? Tscal{0} : in[id];
                            });
                    });
            });
    }

    template<class Tvec, class TgridVec>
    std::string NodeSpMVPoisson3D<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Poisson sparse matrix-vector product (7-point finite volume stencil)
            \begin{equation}
            {spans_out}_i = - \sum_{f \in \partial i} S_f \frac{{spans_in}_{j(f)} - {spans_in}_i}{d_{ij}}
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSpMVPoisson3D<f64_3, i64_3>;

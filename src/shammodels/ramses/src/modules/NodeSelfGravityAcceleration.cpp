// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSelfGravityAcceleration.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Self-gravity acceleration \f$ \mathbf{g} = -\nabla \phi \f$ on the AMR grid.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/NodeSelfGravityAcceleration.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include <array>

namespace {

    using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
    using Direction            = shammodels::basegodunov::modules::Direction;

    /// Get \f$ -\nabla \phi \f$ of a cell, see NodeSelfGravityAcceleration
    template<class Tscal, class ACCField>
    inline std::array<Tscal, 3> get_3d_minus_phi_grad(
        const Tscal *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {

        auto cur_cell_block_id = cell_global_id / block_size;

        auto get_gradient_dir = [&](auto &graph_links, Direction dir) -> Tscal {
            Tscal acc             = 0;
            auto cell_center_dist = cell_sizes[cur_cell_block_id];
            Tscal fac             = 1.;
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;

                int sign = 1 - 2 * (dir % 2);
                acc += sign * (field_access(id_b) - field_access(cell_global_id));

                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 2.);
                }
                // This logic supposes that the last (4-th) cell at the interface has the same size
                // as the other three cells. This is consistent with 2:1 refinement.
                // TODO: extend to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
            });
            return (cnt > 0) ? acc / (cell_center_dist * fac * cnt) : Tscal{0};
        };

        Tscal delta_xp = get_gradient_dir(graph_iter_xp, Direction::xp);
        Tscal delta_xm = get_gradient_dir(graph_iter_xm, Direction::xm);
        Tscal delta_yp = get_gradient_dir(graph_iter_yp, Direction::yp);
        Tscal delta_ym = get_gradient_dir(graph_iter_ym, Direction::ym);
        Tscal delta_zp = get_gradient_dir(graph_iter_zp, Direction::zp);
        Tscal delta_zm = get_gradient_dir(graph_iter_zm, Direction::zm);

        Tscal phi_gx = -0.5 * (delta_xp + delta_xm);
        Tscal phi_gy = -0.5 * (delta_yp + delta_ym);
        Tscal phi_gz = -0.5 * (delta_zp + delta_zm);

        return {phi_gx, phi_gy, phi_gz};
    }

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeSelfGravityAcceleration<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_phi_g.ensure_sizes(edges.sizes.indexes);

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.cell_neigh_graph.graph.for_each(
            [&](u64 id, const OrientedAMRGraph<Tvec, TgridVec> &oriented_cell_graph) {
                auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                auto &phi_span        = edges.spans_phi.get_spans().get(id);
                auto &phi_g_span      = edges.spans_phi_g.get_spans().get(id);

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
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm},
                    sham::MultiRef{phi_g_span},
                    cell_count,
                    [block_size = block_size](
                        u32 cell_global_id,
                        const Tscal *__restrict cell_sizes,
                        const Tscal *__restrict phi,
                        const AMRGraph::ro_access graph_iter_xp,
                        const AMRGraph::ro_access graph_iter_xm,
                        const AMRGraph::ro_access graph_iter_yp,
                        const AMRGraph::ro_access graph_iter_ym,
                        const AMRGraph::ro_access graph_iter_zp,
                        const AMRGraph::ro_access graph_iter_zm,
                        Tvec *__restrict phi_g) {
                        auto g = get_3d_minus_phi_grad<Tscal>(
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
                        phi_g[cell_global_id] = {g[0], g[1], g[2]};
                    });
            });
    }

    template<class Tvec, class TgridVec>
    std::string NodeSelfGravityAcceleration<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Self-gravity acceleration
            \begin{equation}
            {spans_phi_g}_i = -(\nabla {spans_phi})_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSelfGravityAcceleration<f64_3, i64_3>;

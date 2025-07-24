// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGMatVecProd.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief Implementation of matrix-vector product [A*p] for the conjugate gradient solver.
 *
 */
#include "shambase/string.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shammodels/ramses/modules/CGMatVecProd.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <type_traits>

namespace {

    template<class Tvec, class TgridVec>
    class _Kernel {
        using Tscal            = shambase::VecComponent<Tvec>;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges =
            typename shammodels::basegodunov::modules::NodeCGMatVecProd<Tvec, TgridVec>::Edges;

        public:
        inline static void kernel(Edges &edges, u32 block_size) {
            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                    auto &phi_p_span      = edges.spans_phi_p.get_spans().get(id);
                    auto &phi_Ap_span     = edges.spans_phi_Ap.get_spans().get(id);

                    AMRGraph &graph_neigh_xp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xp]);
                    AMRGraph &graph_neigh_xm
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xm]);
                    AMRGraph &graph_neigh_yp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::yp]);
                    AMRGraph &graph_neigh_ym
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::ym]);
                    AMRGraph &graph_neigh_zp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zp]);
                    AMRGraph &graph_neigh_zm
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zm]);

                    sham::EventList depends_list;

                    auto cell_sizes = cell_sizes_span.get_read_access(depends_list);
                    auto phi_p      = phi_p_span.get_read_access(depends_list);
                    auto phi_Ap     = phi_Ap_span.get_write_access(depends_list);

                    auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
                    auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
                    auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
                    auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
                    auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
                    auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                        u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                        shambase::parralel_for(cgh, cell_count, "compute Ap mat-vec", [=](u64 gid) {
                            const u32 cell_global_id = (u32) gid;
                            const u32 block_id       = cell_global_id / block_size;
                            const u32 cell_loc_id    = cell_global_id % block_size;

                            Tscal delta_cell = cell_sizes[block_id];
                            auto Ap_id       = laplacian_stencil_id<Tscal, Tvec>(
                                cell_global_id,
                                delta_cell,
                                graph_iter_xp,
                                graph_iter_xm,
                                graph_iter_yp,
                                graph_iter_ym,
                                graph_iter_zp,
                                graph_iter_zm,
                                [=](u32 id) {
                                    return phi_p[id];
                                });
                            phi_Ap[cell_global_id] = Ap_id;
                        });
                    });

                    cell_sizes_span.complete_event_state(e);
                    phi_p_span.complete_event_state(e);
                    phi_Ap_span.complete_event_state(e);

                    graph_neigh_xp.complete_event_state(e);
                    graph_neigh_xm.complete_event_state(e);
                    graph_neigh_yp.complete_event_state(e);
                    graph_neigh_ym.complete_event_state(e);
                    graph_neigh_zp.complete_event_state(e);
                    graph_neigh_zm.complete_event_state(e);
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeCGMatVecProd<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi_p.check_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.check_sizes(edges.sizes.indexes);

        _Kernel<Tvec, TgridVec>::kernel(edges, block_size);
    }

    template<class Tvec, class TgridVec>
    std::string NodeCGMatVecProd<Tvec, TgridVec>::_impl_get_tex() {

        std::string span_phi_p  = get_ro_edge_base(3).get_tex_symbol();
        std::string span_phi_Ap = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute Ap matrix-vector product
            \begin{equation}
            \mathbf{result} = \mathbf{A}\mathbf{p}
            \end{equation}
        )tex";

        shambase::replace_all(tex, "{result}", span_phi_Ap);
        shambase::replace_all(tex, "{p}", span_phi_p);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGMatVecProd<f64_3, i64_3>;

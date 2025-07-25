// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGInit.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/CGInit.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <type_traits>

namespace {
    template<class Tvec, class TgridVec>
    class _Kernel {
        using Tscal     = shambase::VecComponent<Tvec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges = typename shammodels::basegodunov::modules::CGInit<Tvec, TgridVec>::Edges;

        public:
        inline static void kernel(Edges &edges, u32 block_size, Tscal fourPiG) {
            edges.cell_neigh_graph.graph.for_each([&](u64 id,
                                                      const OrientedAMRGraph &oriented_cell_graph) {
                auto &block_level_span = edges.spans_block_level.get_spans().get(id);
                auto &block_max_span   = edges.spans_block_max.get_spans().get(id);
                auto &block_min_span   = edges.spans_block_min.get_spans().get(id);
                auto &phi_span         = edges.spans_phi.get_spans().get(id);
                auto &rho_span         = edges.spans_rho.get_spans().get(id);
                auto &mean_rho         = edges.mean_rho.value;
                auto &phi_res_span     = edges.spans_phi_res.get_spans().get(id);
                auto &phi_p_span       = edges.spans_phi_p.get_spans().get(id);

                AMRGraph &graph_neigh_xp = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::xp]);
                AMRGraph &graph_neigh_xm = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::xm]);
                AMRGraph &graph_neigh_yp = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::yp]);
                AMRGraph &graph_neigh_ym = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::ym]);
                AMRGraph &graph_neigh_zp = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::zp]);
                AMRGraph &graph_neigh_zm = shambase::get_check_ref(
                    oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::zm]);

                u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        block_min_span,
                        block_max_span,
                        block_level_span,
                        phi_span,
                        rho_span,
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm},
                    sham::MultiRef{phi_res_span, phi_p_span},
                    cell_count,
                    [=](i32 cell_global_id,
                        const TgridVec *__restrict block_min,
                        const TgridVec *__restrict block_max,
                        const TgridUint *__restrict block_level,
                        const Tscal *__restrict phi,
                        const Tscal *__restrict rho,
                        const auto graph_iter_xp,
                        const auto graph_iter_xm,
                        const auto graph_iter_yp,
                        const auto graph_iter_ym,
                        const auto graph_iter_zp,
                        const auto graph_iter_zm,
                        Tscal *__restrict phi_res,
                        Tscal *__restrict phi_p) {
                        auto Aphi = shammodels::basegodunov::laplacian_7pt<Tscal, Tvec, TgridUint>(
                            cell_global_id,
                            graph_iter_xp,
                            graph_iter_xm,
                            graph_iter_yp,
                            graph_iter_ym,
                            graph_iter_zp,
                            graph_iter_zm,
                            [=](u32 id) {
                                return phi[id];
                            },

                            [=](u32 id) {
                                return block_level[id];
                            },

                            [=](u32 id) {
                                return block_min[id];
                            },

                            [=](u32 id) {
                                return block_max[id];
                            });

                        auto res                = fourPiG * (rho[cell_global_id] - mean_rho) - Aphi;
                        phi_res[cell_global_id] = res;
                        phi_p[cell_global_id]   = res;
                    });
            });
        }
    };
} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void CGInit<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_level.check_sizes(edges.sizes.indexes);
        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.check_sizes(edges.sizes.indexes);
        edges.spans_phi_p.check_sizes(edges.sizes.indexes);

        _Kernel<Tvec, TgridVec>::kernel(edges, block_size, fourPiG);
    }

    template<class Tvec, class TgridVec>
    std::string CGInit<Tvec, TgridVec>::_impl_get_tex() {
        std::string sizes                  = get_ro_edge_base(0).get_tex_symbol();
        std::string cell_neigh_graph       = get_ro_edge_base(1).get_tex_symbol();
        std::string spans_block_cell_sizes = get_ro_edge_base(2).get_tex_symbol();
        std::string span_phi               = get_ro_edge_base(3).get_tex_symbol();
        std::string span_rho               = get_ro_edge_base(4).get_tex_symbol();
        std::string mean_rho               = get_ro_edge_base(5).get_tex_symbol();
        std::string span_phi_res           = get_rw_edge_base(0).get_tex_symbol();
        std::string span_phi_p             = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Initiation step of CG
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);
        shambase::replace_all(tex, "{spans_block_cell_sizes}", spans_block_cell_sizes);
        shambase::replace_all(tex, "{span_phi}", span_phi);
        shambase::replace_all(tex, "{span_rhi}", span_rho);
        shambase::replace_all(tex, "{mean_rho}", mean_rho);
        shambase::replace_all(tex, "{span_phi_res}", span_phi_res);
        shambase::replace_all(tex, "{span_phi_p}", span_phi_p);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::CGInit<f64_3, i64_3>;

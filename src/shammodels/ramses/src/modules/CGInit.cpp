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
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <type_traits>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

namespace {
    using Direction = shammodels::basegodunov::modules::Direction;

    /**
     * @brief Get the discretized laplacian
     *
     * @tparam T
     * @tparam Tvec
     * @tparam ACCField
     * @param cell_global_id
     * @param delta_cell
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access
     * @return T
     */
    template<class T, class Tvec, class ACCField>
    inline T get_gemv_id(
        const u32 cell_global_id,
        const T delta_cell,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {

        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        T W_i  = field_access(cell_global_id);
        T W_xp = get_avg_neigh(graph_iter_xp);
        T W_xm = get_avg_neigh(graph_iter_xm);
        T W_yp = get_avg_neigh(graph_iter_yp);
        T W_ym = get_avg_neigh(graph_iter_ym);
        T W_zp = get_avg_neigh(graph_iter_zp);
        T W_zm = get_avg_neigh(graph_iter_zm);

        T inv_delta_cell_sqr = 1.0 / (delta_cell * delta_cell);

        T laplace_x = inv_delta_cell_sqr * (W_xm - 2. * W_i + W_xp);
        T laplace_y = inv_delta_cell_sqr * (W_ym - 2. * W_i + W_yp);
        T laplace_z = inv_delta_cell_sqr * (W_zm - 2. * W_i + W_zp);

        return (laplace_x + laplace_y + laplace_z);
    }

    template<class Tvec, class TgridVec>
    class _Kernel {
        using Tscal            = shambase::VecComponent<Tvec>;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges = typename shammodels::basegodunov::modules::CGInit<Tvec, TgridVec>::Edges;

        public:
        inline static void kernel(Edges &edges, u32 block_size, Tscal fourPiG) {
            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                    auto &phi_span        = edges.spans_phi.get_spans().get(id);
                    auto &rho_span        = edges.spans_rho.get_spans().get(id);
                    auto &mean_rho        = edges.mean_rho.value;
                    auto &phi_res_span    = edges.spans_phi_res.get_spans().get(id);
                    auto &phi_p_span      = edges.spans_phi_p.get_spans().get(id);

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
                    auto phi        = phi_span.get_read_access(depends_list);
                    auto rho        = rho_span.get_read_access(depends_list);
                    auto phi_res    = phi_res_span.get_write_access(depends_list);
                    auto phi_p      = phi_p_span.get_write_access(depends_list);

                    auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
                    auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
                    auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
                    auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
                    auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
                    auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                        u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                        shambase::parralel_for(cgh, cell_count, "init step for cg", [=](u64 gid) {
                            const u32 cell_global_id = (u32) gid;
                            const u32 block_id       = cell_global_id / block_size;
                            const u32 cell_loc_id    = cell_global_id % block_size;

                            Tscal delta_cell = cell_sizes[block_id];
                            auto Aphi        = get_gemv_id<Tscal, Tvec>(
                                cell_global_id,
                                delta_cell,
                                graph_iter_xp,
                                graph_iter_xm,
                                graph_iter_yp,
                                graph_iter_ym,
                                graph_iter_zp,
                                graph_iter_zm,
                                [=](u32 id) {
                                    return phi[id];
                                });

                            auto res = fourPiG * (rho[cell_global_id] - mean_rho) - Aphi;
                            phi_res[cell_global_id] = res;
                            phi_p[cell_global_id]   = res;
                        });
                    });

                    cell_sizes_span.complete_event_state(e);
                    phi_span.complete_event_state(e);
                    rho_span.complete_event_state(e);
                    phi_res_span.complete_event_state(e);
                    phi_p_span.complete_event_state(e);

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
    void CGInit<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
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

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeGetNextConsVar.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeGetNextConsVar.hpp"
#include <type_traits>

namespace {
    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec, class TgridVec>
    class KernelGetNextConsVar {
        using Tscal            = sham::VecComponent<Tvec>;
        using TgridUint        = typename std::make_unsigned<sham::VecComponent<TgridVec>>::type;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges =
            typename shammodels::basegodunov::modules::NodeGetNextConsVar<Tvec, TgridVec>::Edges;
        using Config   = shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        public:
        inline static void kernel(Edges &edges, u32 block_size) {

            edges.cell_neigh_graph.graph.for_each([&](u64 id,
                                                      const OrientedAMRGraph &oriented_cell_graph) {
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

                u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                auto &span_flux_rhov_xp
                    = edges.flux_rhov_face_xp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhov_xm
                    = edges.flux_rhov_face_xm.link_fields.get(id).link_graph_field;
                auto &span_flux_rhov_yp
                    = edges.flux_rhov_face_yp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhov_ym
                    = edges.flux_rhov_face_ym.link_fields.get(id).link_graph_field;
                auto &span_flux_rhov_zp
                    = edges.flux_rhov_face_zp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhov_zm
                    = edges.flux_rhov_face_zm.link_fields.get(id).link_graph_field;

                auto &span_flux_rhoe_xp
                    = edges.flux_rhoe_face_xp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhoe_xm
                    = edges.flux_rhoe_face_xm.link_fields.get(id).link_graph_field;
                auto &span_flux_rhoe_yp
                    = edges.flux_rhoe_face_yp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhoe_ym
                    = edges.flux_rhoe_face_ym.link_fields.get(id).link_graph_field;
                auto &span_flux_rhoe_zp
                    = edges.flux_rhoe_face_zp.link_fields.get(id).link_graph_field;
                auto &span_flux_rhoe_zm
                    = edges.flux_rhoe_face_zm.link_fields.get(id).link_graph_field;

                auto &block_lower_span = edges.spans_cell0block_aabb_lower.get_spans().get(id);
                auto &cell_size_span   = edges.spans_block_cell_sizes.get_spans().get(id);

                auto &span_old_rho    = edges.spans_rho_old.get_spans().get(id);
                auto &span_next_rho   = edges.spans_rho_next.get_spans().get(id);
                auto &span_old_rhov   = edges.spans_rhov_old.get_spans().get(id);
                auto &span_old_rhoe   = edges.spans_rhoe_old.get_spans().get(id);
                auto &span_old_phi_g  = edges.spans_phi_g_old.get_spans().get(id);
                auto &span_next_phi_g = edges.spans_phi_g_next.get_spans().get(id);
                auto &span_next_rhov  = edges.spans_rhov_next.get_spans().get(id);
                auto &span_next_rhoe  = edges.spans_rhoe_next.get_spans().get(id);

                auto &dt = edges.dt.value;

                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        cell_size_span,    block_lower_span,  span_flux_rhov_xp, span_flux_rhov_xm,
                        span_flux_rhov_yp, span_flux_rhov_ym, span_flux_rhov_zp, span_flux_rhov_zm,
                        span_flux_rhoe_xp, span_flux_rhoe_xm, span_flux_rhoe_yp, span_flux_rhoe_ym,
                        span_flux_rhoe_zp, span_flux_rhoe_zm, span_old_rho,      span_next_rho,
                        span_old_rhov,     span_old_rhoe,     span_old_phi_g,    span_next_phi_g,

                        graph_neigh_xp,    graph_neigh_xm,    graph_neigh_yp,    graph_neigh_ym,
                        graph_neigh_zp,    graph_neigh_zm},
                    sham::MultiRef{span_next_rhov, span_next_rhoe},
                    cell_count,

                    [block_size, dt](
                        i32 i,
                        const Tscal *__restrict cell_size,
                        const Tvec *__restrict aabb_lower,
                        const auto flux_rhov_xp,
                        const auto flux_rhov_xm,
                        const auto flux_rhov_yp,
                        const auto flux_rhov_ym,
                        const auto flux_rhov_zp,
                        const auto flux_rhov_zm,
                        const auto flux_rhoe_xp,
                        const auto flux_rhoe_xm,
                        const auto flux_rhoe_yp,
                        const auto flux_rhoe_ym,
                        const auto flux_rhoe_zp,
                        const auto flux_rhoe_zm,
                        const Tscal *__restrict old_rho,
                        const Tscal *__restrict next_rho,
                        const Tvec *__restrict old_rhov,
                        const Tscal *__restrict old_rhoe,
                        const Tvec *__restrict old_phi_g,
                        const Tvec *__restrict next_phi_g,
                        const auto graph_iter_xp,
                        const auto graph_iter_xm,
                        const auto graph_iter_yp,
                        const auto graph_iter_ym,
                        const auto graph_iter_zp,
                        const auto graph_iter_zm,
                        Tvec *__restrict next_rhov,
                        Tscal *__restrict next_rhoe) {
                        /**/
                        auto get_cell_aabb = [=](u32 id) -> shammath::AABB<Tvec> {
                            const u32 cell_global_id = (u32) id;

                            const u32 block_id    = cell_global_id / block_size;
                            const u32 cell_loc_id = cell_global_id % block_size;

                            // fetch current block info
                            const Tvec cblock_min  = aabb_lower[block_id];
                            const Tscal delta_cell = cell_size[block_id];

                            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                            Tvec offset
                                = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

                            Tvec aabb_min = cblock_min + offset;
                            Tvec aabb_max = aabb_min + delta_cell;

                            return {aabb_min, aabb_max};
                        };

                        /**/
                        auto get_face_surface = [=](u32 id_a, u32 id_b) -> Tscal {
                            shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
                            shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

                            shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

                            Tvec delta_face = face_aabb.delt();

                            delta_face.x() = (delta_face.x() == 0) ? 1 : delta_face.x();
                            delta_face.y() = (delta_face.y() == 0) ? 1 : delta_face.y();
                            delta_face.z() = (delta_face.z() == 0) ? 1 : delta_face.z();

                            return delta_face.x() * delta_face.y() * delta_face.z();
                        };

                        /**/
                        const u32 block_id    = i / block_size;
                        const u32 cell_loc_id = i % block_size;

                        Tscal V_i = cell_size[block_id];
                        V_i       = V_i * V_i * V_i;

                        Tvec dtrhov  = {0, 0, 0};
                        Tscal dtrhoe = 0;

                        graph_iter_xp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_xp[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_xp[link_id] * S_ij;
                        });
                        graph_iter_xm.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_xm[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_xm[link_id] * S_ij;
                        });
                        graph_iter_yp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_yp[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_yp[link_id] * S_ij;
                        });
                        graph_iter_ym.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_ym[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_ym[link_id] * S_ij;
                        });
                        graph_iter_zp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_zp[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_zp[link_id] * S_ij;
                        });
                        graph_iter_zm.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrhov -= flux_rhov_zm[link_id] * S_ij;
                            dtrhoe -= flux_rhoe_zm[link_id] * S_ij;
                        });

                        dtrhoe /= V_i;
                        dtrhov /= V_i;

                        Tvec tmp
                            = old_rhov[i] + dt * dtrhov
                              - 0.5 * dt
                                    * (old_rho[i] * old_phi_g[i] + next_rho[i] * next_phi_g[i]);
                        next_rhov[i] = tmp;
                        next_rhoe[i]
                            = old_rhoe[i] + dt * dtrhoe
                              - 0.5 * dt
                                    * (old_rho[i]
                                           * sham::dot((old_rhov[i] / old_rho[i]), old_phi_g[i])
                                       + next_rho[i]
                                             * sham::dot((tmp / next_rho[i]), next_phi_g[i]));
                    });
            });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeGetNextConsVar<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        // logger::raw_ln("dt in NodeGetNextConsVar", edges.dt.value);

        if (edges.dt.value != 0) {
            edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
            edges.spans_cell0block_aabb_lower.check_sizes(edges.sizes.indexes);
            edges.spans_rho_old.check_sizes(edges.sizes.indexes);
            edges.spans_rho_next.check_sizes(edges.sizes.indexes);
            edges.spans_rhov_old.check_sizes(edges.sizes.indexes);
            edges.spans_rhoe_old.check_sizes(edges.sizes.indexes);
            edges.spans_phi_g_old.check_sizes(edges.sizes.indexes);
            edges.spans_phi_g_next.check_sizes(edges.sizes.indexes);

            edges.spans_rhov_next.ensure_sizes(edges.sizes.indexes);
            edges.spans_rhoe_next.ensure_sizes(edges.sizes.indexes);

            KernelGetNextConsVar<Tvec, TgridVec>::kernel(edges, block_size);
        } else {
            return;
        }
    }
} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeGetNextConsVar<f64_3, i64_3>;

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
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeSelfGravityAcceleration.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include <type_traits>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

namespace {

    using Direction = shammodels::basegodunov::modules::Direction;

    /**
     * @brief Get the 3d, minus phi's gradient
     *
     * @tparam T
     * @tparam Tvec
     * @tparam ACCField
     * @param cell_global_id
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access
     * @return std::array<T, 3>
     */
    template<class Tscal, class Tvec, class ACCField>
    inline std::array<Tscal, 3> get_3d_phi_grad(
        const f64 *cell_sizes,
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

        auto get_gradiant_dir = [&](auto &graph_links, Direction dir) -> Tscal {
            Tscal acc             = shambase::VectorProperties<Tscal>::get_zero();
            auto cell_center_dist = cell_sizes[cur_cell_block_id];
            auto fac              = 1.;
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;

                int sign = 1 - 2 * (dir % 2);
                acc += sign * (field_access(id_b) - field_access(cell_global_id));

                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 2.);
                }
                // This logic suppose that the last (4-th) cell at interface have same size with the
                // other three cells. This is also consitent with 2:1 refinement.
                // TODO: extended to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
            });
            return (cnt > 0) ? acc / (cell_center_dist * fac * cnt)
                             : shambase::VectorProperties<Tscal>::get_zero();
        };

        Tscal delta_xp = get_gradiant_dir(graph_iter_xp, Direction::xp);
        Tscal delta_xm = get_gradiant_dir(graph_iter_xm, Direction::xm);
        Tscal delta_yp = get_gradiant_dir(graph_iter_yp, Direction::yp);
        Tscal delta_ym = get_gradiant_dir(graph_iter_ym, Direction::ym);
        Tscal delta_zp = get_gradiant_dir(graph_iter_zp, Direction::zp);
        Tscal delta_zm = get_gradiant_dir(graph_iter_zm, Direction::zm);

        Tscal phi_gx = -0.5 * (delta_xp + delta_xm);
        Tscal phi_gy = -0.5 * (delta_yp + delta_ym);
        Tscal phi_gz = -0.5 * (delta_zp + delta_zm);

        return {phi_gx, phi_gy, phi_gz};

        // auto get_avg_neigh = [&](auto &graph_links) -> Tscal {
        //     Tscal acc = shambase::VectorProperties<Tscal>::get_zero();
        //     u32 cnt   = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
        //         acc += field_access(id_b);
        //     });
        //     return (cnt > 0) ? acc / cnt : shambase::VectorProperties<Tscal>::get_zero();
        // };

        // Tscal phi_i  = field_access(cell_global_id);
        // Tscal phi_xp = get_avg_neigh(graph_iter_xp);
        // Tscal phi_xm = get_avg_neigh(graph_iter_xm);
        // Tscal phi_yp = get_avg_neigh(graph_iter_yp);
        // Tscal phi_ym = get_avg_neigh(graph_iter_ym);
        // Tscal phi_zp = get_avg_neigh(graph_iter_zp);
        // Tscal phi_zm = get_avg_neigh(graph_iter_zm);

        // /* this intermediate state is not require*/
        // Tscal delta_phi_x_p = phi_xp - phi_i;
        // Tscal delta_phi_y_p = phi_yp - phi_i;
        // Tscal delta_phi_z_p = phi_zp - phi_i;

        // Tscal delta_phi_x_m = phi_i - phi_xm;
        // Tscal delta_phi_y_m = phi_i - phi_ym;
        // Tscal delta_phi_z_m = phi_i - phi_zm;

        // Tscal fact = 1. / Tscal(delta_cell);

        // Tscal phi_gx = -0.5 * (delta_phi_x_m + delta_phi_x_p) * fact;
        // Tscal phi_gy = -0.5 * (delta_phi_y_m + delta_phi_y_p) * fact;
        // Tscal phi_gz = -0.5 * (delta_phi_z_m + delta_phi_z_p) * fact;

        // return {phi_gx, phi_gy, phi_gz};
    }

    template<class Tvec, class TgridVec>
    class KernelSelfGravAcc {

        using Tscal     = sham::VecComponent<Tvec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges            = typename shammodels::basegodunov::modules::
            NodeSelfGravityAcceleration<Tvec, TgridVec>::Edges;

        public:
        inline static void kernel(Edges &edges, u32 block_size) {

            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &phi_span        = edges.spans_phi.get_spans().get(id);
                    auto &phi_g_span      = edges.spans_phi_g.get_spans().get(id);
                    auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);

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
                        [block_size](
                            i32 cell_global_id,
                            const Tscal *__restrict cell_sizes,
                            const Tscal *__restrict in,
                            const auto graph_iter_xp,
                            const auto graph_iter_xm,
                            const auto graph_iter_yp,
                            const auto graph_iter_ym,
                            const auto graph_iter_zp,
                            const auto graph_iter_zm,
                            Tvec *__restrict out) {
                            auto grad_res = get_3d_phi_grad<Tscal, Tvec>(
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
                                    return in[id];
                                });
                            out[cell_global_id] = {grad_res[0], grad_res[1], grad_res[2]};
                        }

                    );
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeSelfGravityAcceleration<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        {
            edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
            edges.spans_phi.check_sizes(edges.sizes.indexes);
            edges.spans_phi_g.ensure_sizes(edges.sizes.indexes);

            KernelSelfGravAcc<Tvec, TgridVec>::kernel(edges, block_size);
        }
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSelfGravityAcceleration<f64_3, i64_3>;

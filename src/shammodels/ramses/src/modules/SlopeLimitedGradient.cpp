// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SlopeLimitedGradient.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/SlopeLimitedGradient.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/SlopeLimitedGradientUtilities.hpp"
#include <type_traits>

namespace {

    template<class Tvec, class TgridVec, SlopeMode mode>
    class KernelSlopeLimScalGrad {

        using Edges = typename shammodels::basegodunov::modules::
            SlopeLimitedScalarGradient<Tvec, TgridVec>::Edges;
        using Tscal            = shambase::VecComponent<Tvec>;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;

        public:
        inline static void kernel(Edges &edges, u32 block_size, u32 var_per_cell) {

            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &field_span      = edges.span_field.get_spans().get(id);
                    auto &field_grad_span = edges.span_grad_field.get_spans().get(id);
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

                    sham::EventList depends_list;

                    auto cell_sizes = cell_sizes_span.get_read_access(depends_list);
                    auto field      = field_span.get_read_access(depends_list);
                    auto field_grad = field_grad_span.get_write_access(depends_list);

                    auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
                    auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
                    auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
                    auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
                    auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
                    auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                        u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                        shambase::parallel_for(
                            cgh, cell_count * var_per_cell, "compute_grad_rho", [=](u64 gid) {
                                const u32 tmp_gid = (u32) gid;

                                const u32 cell_global_id = tmp_gid / var_per_cell;
                                const u32 var_off_loc    = tmp_gid % var_per_cell;

                                const u32 block_id    = cell_global_id / block_size;
                                const u32 cell_loc_id = cell_global_id % block_size;

                                Tscal delta_cell = cell_sizes[block_id];

                                auto result = get_3d_grad<Tscal, Tvec, mode>(
                                    cell_global_id,
                                    delta_cell,
                                    graph_iter_xp,
                                    graph_iter_xm,
                                    graph_iter_yp,
                                    graph_iter_ym,
                                    graph_iter_zp,
                                    graph_iter_zm,
                                    [=](u32 id) {
                                        return field[var_per_cell * id + var_off_loc];
                                    });

                                field_grad[var_per_cell * cell_global_id + var_off_loc]
                                    = {result[0], result[1], result[2]};
                            });
                    });

                    cell_sizes_span.complete_event_state(e);
                    field_span.complete_event_state(e);
                    field_grad_span.complete_event_state(e);

                    graph_neigh_xp.complete_event_state(e);
                    graph_neigh_xm.complete_event_state(e);
                    graph_neigh_yp.complete_event_state(e);
                    graph_neigh_ym.complete_event_state(e);
                    graph_neigh_zp.complete_event_state(e);
                    graph_neigh_zm.complete_event_state(e);
                });
        }
    };

    template<class Tvec, class TgridVec, SlopeMode mode>
    class KernelSlopeLimVecGrad {

        using Edges = typename shammodels::basegodunov::modules::
            SlopeLimitedVectorGradient<Tvec, TgridVec>::Edges;
        using Tscal            = shambase::VecComponent<Tvec>;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;

        public:
        inline static void kernel(Edges &edges, u32 block_size, u32 var_per_cell) {

            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &field_span      = edges.span_field.get_spans().get(id);
                    auto &field_dx_span   = edges.span_dx_field.get_spans().get(id);
                    auto &field_dy_span   = edges.span_dy_field.get_spans().get(id);
                    auto &field_dz_span   = edges.span_dz_field.get_spans().get(id);
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

                    sham::EventList depends_list;

                    auto cell_sizes = cell_sizes_span.get_read_access(depends_list);
                    auto field      = field_span.get_read_access(depends_list);
                    auto field_dx   = field_dx_span.get_write_access(depends_list);
                    auto field_dy   = field_dy_span.get_write_access(depends_list);
                    auto field_dz   = field_dz_span.get_write_access(depends_list);

                    auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
                    auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
                    auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
                    auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
                    auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
                    auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                        u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                        shambase::parallel_for(
                            cgh, cell_count * var_per_cell, "compute_grad_rho", [=](u64 gid) {
                                const u32 tmp_gid = (u32) gid;

                                const u32 cell_global_id = tmp_gid / var_per_cell;
                                const u32 var_off_loc    = tmp_gid % var_per_cell;

                                const u32 block_id    = cell_global_id / block_size;
                                const u32 cell_loc_id = cell_global_id % block_size;

                                Tscal delta_cell = cell_sizes[block_id];

                                auto result = get_3d_grad<Tvec, Tvec, mode>(
                                    cell_global_id,
                                    delta_cell,
                                    graph_iter_xp,
                                    graph_iter_xm,
                                    graph_iter_yp,
                                    graph_iter_ym,
                                    graph_iter_zp,
                                    graph_iter_zm,
                                    [=](u32 id) {
                                        return field[var_per_cell * id + var_off_loc];
                                    });

                                field_dx[var_per_cell * cell_global_id + var_off_loc] = result[0];
                                field_dy[var_per_cell * cell_global_id + var_off_loc] = result[1];
                                field_dz[var_per_cell * cell_global_id + var_off_loc] = result[2];
                            });
                    });

                    cell_sizes_span.complete_event_state(e);
                    field_span.complete_event_state(e);
                    field_dx_span.complete_event_state(e);
                    field_dy_span.complete_event_state(e);
                    field_dz_span.complete_event_state(e);

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
    void SlopeLimitedScalarGradient<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.span_field.check_sizes(edges.sizes.indexes);

        edges.span_grad_field.ensure_sizes(edges.sizes.indexes);

        if (mode == SlopeMode::None) {
            using Kern = KernelSlopeLimScalGrad<Tvec, TgridVec, None>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_f) {
            using Kern = KernelSlopeLimScalGrad<Tvec, TgridVec, VanLeer_f>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_std) {
            using Kern = KernelSlopeLimScalGrad<Tvec, TgridVec, VanLeer_std>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_sym) {
            using Kern = KernelSlopeLimScalGrad<Tvec, TgridVec, VanLeer_sym>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::Minmod) {
            using Kern = KernelSlopeLimScalGrad<Tvec, TgridVec, Minmod>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, class TgridVec>
    std::string SlopeLimitedScalarGradient<Tvec, TgridVec>::_impl_get_tex() const {

        std::string sizes                  = get_ro_edge_base(0).get_tex_symbol();
        std::string cell_neigh_graph       = get_ro_edge_base(1).get_tex_symbol();
        std::string spans_block_cell_sizes = get_ro_edge_base(2).get_tex_symbol();
        std::string span_field             = get_ro_edge_base(3).get_tex_symbol();
        std::string span_grad_field        = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Slope limited gradient (Scalar)
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);
        shambase::replace_all(tex, "{spans_block_cell_sizes}", spans_block_cell_sizes);
        shambase::replace_all(tex, "{span_field}", span_field);
        shambase::replace_all(tex, "{span_grad_field}", span_grad_field);

        return tex;
    }
    template<class Tvec, class TgridVec>
    void SlopeLimitedVectorGradient<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.span_field.check_sizes(edges.sizes.indexes);

        edges.span_dx_field.ensure_sizes(edges.sizes.indexes);
        edges.span_dy_field.ensure_sizes(edges.sizes.indexes);
        edges.span_dz_field.ensure_sizes(edges.sizes.indexes);

        if (mode == SlopeMode::None) {
            using Kern = KernelSlopeLimVecGrad<Tvec, TgridVec, None>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_f) {
            using Kern = KernelSlopeLimVecGrad<Tvec, TgridVec, VanLeer_f>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_std) {
            using Kern = KernelSlopeLimVecGrad<Tvec, TgridVec, VanLeer_std>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::VanLeer_sym) {
            using Kern = KernelSlopeLimVecGrad<Tvec, TgridVec, VanLeer_sym>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else if (mode == SlopeMode::Minmod) {
            using Kern = KernelSlopeLimVecGrad<Tvec, TgridVec, Minmod>;
            Kern::kernel(edges, block_size, var_per_cell);
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, class TgridVec>
    std::string SlopeLimitedVectorGradient<Tvec, TgridVec>::_impl_get_tex() const {

        std::string sizes                  = get_ro_edge_base(0).get_tex_symbol();
        std::string cell_neigh_graph       = get_ro_edge_base(1).get_tex_symbol();
        std::string spans_block_cell_sizes = get_ro_edge_base(2).get_tex_symbol();
        std::string span_field             = get_ro_edge_base(3).get_tex_symbol();
        std::string span_dx_field          = get_rw_edge_base(0).get_tex_symbol();
        std::string span_dy_field          = get_rw_edge_base(0).get_tex_symbol();
        std::string span_dz_field          = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Slope limited gradient (Vector)
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);
        shambase::replace_all(tex, "{spans_block_cell_sizes}", spans_block_cell_sizes);
        shambase::replace_all(tex, "{span_field}", span_field);
        shambase::replace_all(tex, "{span_dx_field}", span_dx_field);
        shambase::replace_all(tex, "{span_dy_field}", span_dy_field);
        shambase::replace_all(tex, "{span_dz_field}", span_dz_field);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::SlopeLimitedScalarGradient<f64_3, i64_3>;
template class shammodels::basegodunov::modules::SlopeLimitedVectorGradient<f64_3, i64_3>;

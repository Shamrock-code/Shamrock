// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SumFluxHydro.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sum the fluxes into the time derivative fields for Hydro
 *
 */

#include "shammodels/ramses/modules/SumFluxHydro.hpp"

template<class Tvec, class TgridVec>
struct KernelSumFluxHydro {
    using Tscal    = shambase::VecComponent<Tvec>;
    using AMRBlock = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;

    using AMRGraph = shammodels::basegodunov::modules::NeighGraph;

    inline static shammath::AABB<Tvec> get_cell_aabb(
        u32 id,
        const Tvec *__restrict cell0block_aabb_lower,
        const Tscal *__restrict block_cell_sizes) {
        const u32 cell_global_id = (u32) id;

        const u32 block_id    = cell_global_id / AMRBlock::block_size;
        const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

        // fetch current block info
        const Tvec cblock_min  = cell0block_aabb_lower[block_id];
        const Tscal delta_cell = block_cell_sizes[block_id];

        std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
        Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

        Tvec aabb_min = cblock_min + offset;
        Tvec aabb_max = aabb_min + delta_cell;

        return {aabb_min, aabb_max};
    };

    inline static Tscal get_face_surface(
        u32 id_a,
        u32 id_b,
        const Tvec *__restrict cell0block_aabb_lower,
        const Tscal *__restrict block_cell_sizes,
        Tscal dxfact) {
        shammath::AABB<Tvec> aabb_cell_a
            = get_cell_aabb(id_a, cell0block_aabb_lower, block_cell_sizes);
        shammath::AABB<Tvec> aabb_cell_b
            = get_cell_aabb(id_b, cell0block_aabb_lower, block_cell_sizes);

        shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

        Tvec delta_face = face_aabb.delt();

        // smallest possible coordinate delta, anything under this is considered null
        // cell can not have less than size 1 in the grid space, so 1*dxfact is the minimum size, so
        // we check against dxfact/2
        Tscal smd = dxfact / 2;

        delta_face.x() = (delta_face.x() < smd) ? 1 : delta_face.x();
        delta_face.y() = (delta_face.y() < smd) ? 1 : delta_face.y();
        delta_face.z() = (delta_face.z() < smd) ? 1 : delta_face.z();

        return delta_face.x() * delta_face.y() * delta_face.z();
    };

    Tscal dxfact;

    inline void operator()(
        u32 id_a,
        const Tscal *__restrict flux_rho_face_xp,
        const Tscal *__restrict flux_rho_face_xm,
        const Tscal *__restrict flux_rho_face_yp,
        const Tscal *__restrict flux_rho_face_ym,
        const Tscal *__restrict flux_rho_face_zp,
        const Tscal *__restrict flux_rho_face_zm,
        const Tvec *__restrict flux_rhov_face_xp,
        const Tvec *__restrict flux_rhov_face_xm,
        const Tvec *__restrict flux_rhov_face_yp,
        const Tvec *__restrict flux_rhov_face_ym,
        const Tvec *__restrict flux_rhov_face_zp,
        const Tvec *__restrict flux_rhov_face_zm,
        const Tscal *__restrict flux_rhoe_face_xp,
        const Tscal *__restrict flux_rhoe_face_xm,
        const Tscal *__restrict flux_rhoe_face_yp,
        const Tscal *__restrict flux_rhoe_face_ym,
        const Tscal *__restrict flux_rhoe_face_zp,
        const Tscal *__restrict flux_rhoe_face_zm,
        const Tscal *__restrict block_cell_sizes,
        const Tvec *__restrict cell0block_aabb_lower,
        const AMRGraph::ro_access graph_iter_xp,
        const AMRGraph::ro_access graph_iter_xm,
        const AMRGraph::ro_access graph_iter_yp,
        const AMRGraph::ro_access graph_iter_ym,
        const AMRGraph::ro_access graph_iter_zp,
        const AMRGraph::ro_access graph_iter_zm,
        Tscal *__restrict dt_rho,
        Tvec *__restrict dt_rhov,
        Tscal *__restrict dt_rhoe) const {
        const u32 block_id    = id_a / AMRBlock::block_size;
        const u32 cell_loc_id = id_a % AMRBlock::block_size;

        Tscal V_i = block_cell_sizes[block_id];
        V_i       = V_i * V_i * V_i;

        Tscal dtrho  = 0;
        Tvec dtrhov  = {0, 0, 0};
        Tscal dtrhoe = 0;

        auto add_flux = [&](const auto &graph_iter,
                            const Tscal *flux_rho,
                            const Tvec *flux_rhov,
                            const Tscal *flux_rhoe) {
            graph_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                Tscal S_ij = KernelSumFluxHydro<Tvec, TgridVec>::get_face_surface(
                    id_a, id_b, cell0block_aabb_lower, block_cell_sizes, dxfact);
                dtrho -= flux_rho[link_id] * S_ij;
                dtrhov -= flux_rhov[link_id] * S_ij;
                dtrhoe -= flux_rhoe[link_id] * S_ij;
            });
        };

        add_flux(graph_iter_xp, flux_rho_face_xp, flux_rhov_face_xp, flux_rhoe_face_xp);
        add_flux(graph_iter_xm, flux_rho_face_xm, flux_rhov_face_xm, flux_rhoe_face_xm);
        add_flux(graph_iter_yp, flux_rho_face_yp, flux_rhov_face_yp, flux_rhoe_face_yp);
        add_flux(graph_iter_ym, flux_rho_face_ym, flux_rhov_face_ym, flux_rhoe_face_ym);
        add_flux(graph_iter_zp, flux_rho_face_zp, flux_rhov_face_zp, flux_rhoe_face_zp);
        add_flux(graph_iter_zm, flux_rho_face_zm, flux_rhov_face_zm, flux_rhoe_face_zm);

        dtrho /= V_i;
        dtrhov /= V_i;
        dtrhoe /= V_i;

        dt_rho[id_a]  = dtrho;
        dt_rhov[id_a] = dtrhov;
        dt_rhoe[id_a] = dtrhoe;
    }
};

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::NodeSumFluxHydro<Tvec, TgridVec>::_impl_evaluate_internal() {
    auto edges = get_edges();

    edges.spans_dtrho.ensure_sizes(edges.block_counts.indexes);
    edges.spans_dtrhov.ensure_sizes(edges.block_counts.indexes);
    edges.spans_dtrhoe.ensure_sizes(edges.block_counts.indexes);

    auto get_graph_neigh = [&](Direction dir, u64 patch_id) -> const AMRGraph & {
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;

        const CellGraphEdge &cell_neigh_graph                    = edges.cell_neigh_graph;
        const shambase::DistributedData<OrientedAMRGraph> &graph = cell_neigh_graph.graph;
        const OrientedAMRGraph &oriented_cell_graph              = graph.get(patch_id);
        const std::unique_ptr<AMRGraph> &graph_link = oriented_cell_graph.graph_links[dir];

        return shambase::get_check_ref(graph_link);
    };

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    edges.block_counts.indexes.for_each([&](u64 patch_id, const u64 &block_count) {
        const AMRGraph &graph_neigh_xp = get_graph_neigh(Direction::xp, patch_id);
        const AMRGraph &graph_neigh_xm = get_graph_neigh(Direction::xm, patch_id);
        const AMRGraph &graph_neigh_yp = get_graph_neigh(Direction::yp, patch_id);
        const AMRGraph &graph_neigh_ym = get_graph_neigh(Direction::ym, patch_id);
        const AMRGraph &graph_neigh_zp = get_graph_neigh(Direction::zp, patch_id);
        const AMRGraph &graph_neigh_zm = get_graph_neigh(Direction::zm, patch_id);

        auto get_face_buf = [&](auto &neigh_link_field) -> const auto & {
            return neigh_link_field.link_fields.get(patch_id).link_graph_field;
        };

        const auto &buf_flux_rho_face_xp  = get_face_buf(edges.flux_rho_face_xp);
        const auto &buf_flux_rho_face_xm  = get_face_buf(edges.flux_rho_face_xm);
        const auto &buf_flux_rho_face_yp  = get_face_buf(edges.flux_rho_face_yp);
        const auto &buf_flux_rho_face_ym  = get_face_buf(edges.flux_rho_face_ym);
        const auto &buf_flux_rho_face_zp  = get_face_buf(edges.flux_rho_face_zp);
        const auto &buf_flux_rho_face_zm  = get_face_buf(edges.flux_rho_face_zm);
        const auto &buf_flux_rhov_face_xp = get_face_buf(edges.flux_rhov_face_xp);
        const auto &buf_flux_rhov_face_xm = get_face_buf(edges.flux_rhov_face_xm);
        const auto &buf_flux_rhov_face_yp = get_face_buf(edges.flux_rhov_face_yp);
        const auto &buf_flux_rhov_face_ym = get_face_buf(edges.flux_rhov_face_ym);
        const auto &buf_flux_rhov_face_zp = get_face_buf(edges.flux_rhov_face_zp);
        const auto &buf_flux_rhov_face_zm = get_face_buf(edges.flux_rhov_face_zm);
        const auto &buf_flux_rhoe_face_xp = get_face_buf(edges.flux_rhoe_face_xp);
        const auto &buf_flux_rhoe_face_xm = get_face_buf(edges.flux_rhoe_face_xm);
        const auto &buf_flux_rhoe_face_yp = get_face_buf(edges.flux_rhoe_face_yp);
        const auto &buf_flux_rhoe_face_ym = get_face_buf(edges.flux_rhoe_face_ym);
        const auto &buf_flux_rhoe_face_zp = get_face_buf(edges.flux_rhoe_face_zp);
        const auto &buf_flux_rhoe_face_zm = get_face_buf(edges.flux_rhoe_face_zm);

        auto &block_cell_sizes      = edges.spans_block_cell_sizes.get_spans().get(patch_id);
        auto &cell0block_aabb_lower = edges.spans_cell0block_aabb_lower.get_spans().get(patch_id);

        auto &dt_rho_patch  = edges.spans_dtrho.get_spans().get(patch_id);
        auto &dt_rhov_patch = edges.spans_dtrhov.get_spans().get(patch_id);
        auto &dt_rhoe_patch = edges.spans_dtrhoe.get_spans().get(patch_id);

        u32 cell_count = block_count * AMRBlock::block_size;

        sham::kernel_call(
            q,
            sham::MultiRef{buf_flux_rho_face_xp,  buf_flux_rho_face_xm,  buf_flux_rho_face_yp,
                           buf_flux_rho_face_ym,  buf_flux_rho_face_zp,  buf_flux_rho_face_zm,
                           buf_flux_rhov_face_xp, buf_flux_rhov_face_xm, buf_flux_rhov_face_yp,
                           buf_flux_rhov_face_ym, buf_flux_rhov_face_zp, buf_flux_rhov_face_zm,
                           buf_flux_rhoe_face_xp, buf_flux_rhoe_face_xm, buf_flux_rhoe_face_yp,
                           buf_flux_rhoe_face_ym, buf_flux_rhoe_face_zp, buf_flux_rhoe_face_zm,
                           block_cell_sizes,      cell0block_aabb_lower, graph_neigh_xp,
                           graph_neigh_xm,        graph_neigh_yp,        graph_neigh_ym,
                           graph_neigh_zp,        graph_neigh_zm},
            sham::MultiRef{dt_rho_patch, dt_rhov_patch, dt_rhoe_patch},
            cell_count,
            KernelSumFluxHydro<Tvec, TgridVec>{dxfact});
    });
}

template class shammodels::basegodunov::modules::NodeSumFluxHydro<f64_3, i64_3>;

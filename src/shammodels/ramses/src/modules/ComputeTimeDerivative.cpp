// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeTimeDerivative.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeTimeDerivative.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class T>
using NGLink = shammodels::basegodunov::modules::NeighGraphLinkField<T>;

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeTimeDerivative<Tvec, TgridVec>::compute_dt_fields() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> cfield_dtrho
        = utility.make_compute_field<Tscal>("dt rho", AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_dtrhov
        = utility.make_compute_field<Tvec>("dt rhovel", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_dtrhoe
        = utility.make_compute_field<Tscal>("dt rhoe", AMRBlock::block_size);

    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_xp  = storage.flux_rho_face_xp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_xm  = storage.flux_rho_face_xm.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_yp  = storage.flux_rho_face_yp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_ym  = storage.flux_rho_face_ym.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_zp  = storage.flux_rho_face_zp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_face_zm  = storage.flux_rho_face_zm.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_xp  = storage.flux_rhov_face_xp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_xm  = storage.flux_rhov_face_xm.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_yp  = storage.flux_rhov_face_yp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_ym  = storage.flux_rhov_face_ym.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_zp  = storage.flux_rhov_face_zp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_face_zm  = storage.flux_rhov_face_zm.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_xp = storage.flux_rhoe_face_xp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_xm = storage.flux_rhoe_face_xm.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_yp = storage.flux_rhoe_face_yp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_ym = storage.flux_rhoe_face_ym.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_zp = storage.flux_rhoe_face_zp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rhoe_face_zm = storage.flux_rhoe_face_zm.get();

    scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR Flux]", "accumulate fluxes patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        OrientedAMRGraph &oriented_cell_graph
            = shambase::get_check_ref(storage.cell_graph_edge).graph.get(id);

        NGLink<Tscal> &patch_flux_rho_face_xp = flux_rho_face_xp.get(id);
        NGLink<Tscal> &patch_flux_rho_face_xm = flux_rho_face_xm.get(id);
        NGLink<Tscal> &patch_flux_rho_face_yp = flux_rho_face_yp.get(id);
        NGLink<Tscal> &patch_flux_rho_face_ym = flux_rho_face_ym.get(id);
        NGLink<Tscal> &patch_flux_rho_face_zp = flux_rho_face_zp.get(id);
        NGLink<Tscal> &patch_flux_rho_face_zm = flux_rho_face_zm.get(id);

        NGLink<Tvec> &patch_flux_rhov_face_xp = flux_rhov_face_xp.get(id);
        NGLink<Tvec> &patch_flux_rhov_face_xm = flux_rhov_face_xm.get(id);
        NGLink<Tvec> &patch_flux_rhov_face_yp = flux_rhov_face_yp.get(id);
        NGLink<Tvec> &patch_flux_rhov_face_ym = flux_rhov_face_ym.get(id);
        NGLink<Tvec> &patch_flux_rhov_face_zp = flux_rhov_face_zp.get(id);
        NGLink<Tvec> &patch_flux_rhov_face_zm = flux_rhov_face_zm.get(id);

        NGLink<Tscal> &patch_flux_rhoe_face_xp = flux_rhoe_face_xp.get(id);
        NGLink<Tscal> &patch_flux_rhoe_face_xm = flux_rhoe_face_xm.get(id);
        NGLink<Tscal> &patch_flux_rhoe_face_yp = flux_rhoe_face_yp.get(id);
        NGLink<Tscal> &patch_flux_rhoe_face_ym = flux_rhoe_face_ym.get(id);
        NGLink<Tscal> &patch_flux_rhoe_face_zp = flux_rhoe_face_zp.get(id);
        NGLink<Tscal> &patch_flux_rhoe_face_zm = flux_rhoe_face_zm.get(id);

        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_xp  = patch_flux_rho_face_xp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_xm  = patch_flux_rho_face_xm.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_yp  = patch_flux_rho_face_yp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_ym  = patch_flux_rho_face_ym.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_zp  = patch_flux_rho_face_zp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_face_zm  = patch_flux_rho_face_zm.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_xp  = patch_flux_rhov_face_xp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_xm  = patch_flux_rhov_face_xm.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_yp  = patch_flux_rhov_face_yp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_ym  = patch_flux_rhov_face_ym.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_zp  = patch_flux_rhov_face_zp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_face_zm  = patch_flux_rhov_face_zm.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_xp = patch_flux_rhoe_face_xp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_xm = patch_flux_rhoe_face_xm.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_yp = patch_flux_rhoe_face_yp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_ym = patch_flux_rhoe_face_ym.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_zp = patch_flux_rhoe_face_zp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rhoe_face_zm = patch_flux_rhoe_face_zm.link_graph_field;

        sham::DeviceBuffer<Tscal> &dt_rho_patch  = cfield_dtrho.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_patch  = cfield_dtrhov.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &dt_rhoe_patch = cfield_dtrhoe.get_buf_check(id);

        AMRGraph &graph_neigh_xp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        sham::DeviceBuffer<Tscal> &block_cell_sizes
            = shambase::get_check_ref(storage.block_cell_sizes).get_refs().get(id).get().get_buf();
        sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
            = shambase::get_check_ref(storage.cell0block_aabb_lower)
                  .get_refs()
                  .get(id)
                  .get()
                  .get_buf();

        sham::EventList depends_list;
        auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
        auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
        auto acc_dt_rho_patch     = dt_rho_patch.get_write_access(depends_list);
        auto acc_dt_rhov_patch    = dt_rhov_patch.get_write_access(depends_list);
        auto acc_dt_rhoe_patch    = dt_rhoe_patch.get_write_access(depends_list);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        auto flux_rho_face_xp = buf_flux_rho_face_xp.get_read_access(depends_list);
        auto flux_rho_face_xm = buf_flux_rho_face_xm.get_read_access(depends_list);
        auto flux_rho_face_yp = buf_flux_rho_face_yp.get_read_access(depends_list);
        auto flux_rho_face_ym = buf_flux_rho_face_ym.get_read_access(depends_list);
        auto flux_rho_face_zp = buf_flux_rho_face_zp.get_read_access(depends_list);
        auto flux_rho_face_zm = buf_flux_rho_face_zm.get_read_access(depends_list);

        auto flux_rhov_face_xp = buf_flux_rhov_face_xp.get_read_access(depends_list);
        auto flux_rhov_face_xm = buf_flux_rhov_face_xm.get_read_access(depends_list);
        auto flux_rhov_face_yp = buf_flux_rhov_face_yp.get_read_access(depends_list);
        auto flux_rhov_face_ym = buf_flux_rhov_face_ym.get_read_access(depends_list);
        auto flux_rhov_face_zp = buf_flux_rhov_face_zp.get_read_access(depends_list);
        auto flux_rhov_face_zm = buf_flux_rhov_face_zm.get_read_access(depends_list);

        auto flux_rhoe_face_xp = buf_flux_rhoe_face_xp.get_read_access(depends_list);
        auto flux_rhoe_face_xm = buf_flux_rhoe_face_xm.get_read_access(depends_list);
        auto flux_rhoe_face_yp = buf_flux_rhoe_face_yp.get_read_access(depends_list);
        auto flux_rhoe_face_ym = buf_flux_rhoe_face_ym.get_read_access(depends_list);
        auto flux_rhoe_face_zp = buf_flux_rhoe_face_zp.get_read_access(depends_list);
        auto flux_rhoe_face_zm = buf_flux_rhoe_face_zm.get_read_access(depends_list);

        auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
        auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
        auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
        auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
        auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
        auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            auto get_cell_aabb = [=](u32 id) -> shammath::AABB<Tvec> {
                const u32 cell_global_id = (u32) id;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const Tvec cblock_min  = acc_aabb_block_lower[block_id];
                const Tscal delta_cell = acc_aabb_cell_size[block_id];

                std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

                Tvec aabb_min = cblock_min + offset;
                Tvec aabb_max = aabb_min + delta_cell;

                return {aabb_min, aabb_max};
            };

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

            shambase::parralel_for(cgh, cell_count, "accumulate fluxes", [=](u32 id_a) {
                const u32 cell_global_id = (u32) id_a;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                Tscal V_i = acc_aabb_cell_size[block_id];
                V_i       = V_i * V_i * V_i;

                Tscal dtrho  = 0;
                Tvec dtrhov  = {0, 0, 0};
                Tscal dtrhoe = 0;

                graph_iter_xp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_xp[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_xp[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_xp[link_id] * S_ij;
                });

                graph_iter_yp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_yp[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_yp[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_yp[link_id] * S_ij;
                });

                graph_iter_zp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_zp[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_zp[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_zp[link_id] * S_ij;
                });

                graph_iter_xm.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_xm[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_xm[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_xm[link_id] * S_ij;
                });

                graph_iter_ym.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_ym[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_ym[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_ym[link_id] * S_ij;
                });

                graph_iter_zm.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_zm[link_id] * S_ij;
                    dtrhov -= flux_rhov_face_zm[link_id] * S_ij;
                    dtrhoe -= flux_rhoe_face_zm[link_id] * S_ij;
                });

                dtrho /= V_i;
                dtrhov /= V_i;
                dtrhoe /= V_i;

                acc_dt_rho_patch[id_a]  = dtrho;
                acc_dt_rhov_patch[id_a] = dtrhov;
                acc_dt_rhoe_patch[id_a] = dtrhoe;
            });
        });

        cell0block_aabb_lower.complete_event_state(e);
        block_cell_sizes.complete_event_state(e);
        dt_rho_patch.complete_event_state(e);
        dt_rhov_patch.complete_event_state(e);
        dt_rhoe_patch.complete_event_state(e);

        buf_flux_rho_face_xp.complete_event_state(e);
        buf_flux_rho_face_xm.complete_event_state(e);
        buf_flux_rho_face_yp.complete_event_state(e);
        buf_flux_rho_face_ym.complete_event_state(e);
        buf_flux_rho_face_zp.complete_event_state(e);
        buf_flux_rho_face_zm.complete_event_state(e);

        buf_flux_rhov_face_xp.complete_event_state(e);
        buf_flux_rhov_face_xm.complete_event_state(e);
        buf_flux_rhov_face_yp.complete_event_state(e);
        buf_flux_rhov_face_ym.complete_event_state(e);
        buf_flux_rhov_face_zp.complete_event_state(e);
        buf_flux_rhov_face_zm.complete_event_state(e);

        buf_flux_rhoe_face_xp.complete_event_state(e);
        buf_flux_rhoe_face_xm.complete_event_state(e);
        buf_flux_rhoe_face_yp.complete_event_state(e);
        buf_flux_rhoe_face_ym.complete_event_state(e);
        buf_flux_rhoe_face_zp.complete_event_state(e);
        buf_flux_rhoe_face_zm.complete_event_state(e);

        graph_neigh_xp.complete_event_state(e);
        graph_neigh_xm.complete_event_state(e);
        graph_neigh_yp.complete_event_state(e);
        graph_neigh_ym.complete_event_state(e);
        graph_neigh_zp.complete_event_state(e);
        graph_neigh_zm.complete_event_state(e);
    });

    storage.dtrho.set(std::move(cfield_dtrho));
    storage.dtrhov.set(std::move(cfield_dtrhov));
    storage.dtrhoe.set(std::move(cfield_dtrhoe));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeTimeDerivative<Tvec, TgridVec>::
    compute_dt_dust_fields() {
    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    u32 ndust = solver_config.dust_config.ndust;
    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> cfield_dtrho_dust
        = utility.make_compute_field<Tscal>("dt rho dust", ndust * AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_dtrhov_dust
        = utility.make_compute_field<Tvec>("dt rhovel dust", ndust * AMRBlock::block_size);

    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_xp
        = storage.flux_rho_dust_face_xp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_xm
        = storage.flux_rho_dust_face_xm.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_yp
        = storage.flux_rho_dust_face_yp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_ym
        = storage.flux_rho_dust_face_ym.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_zp
        = storage.flux_rho_dust_face_zp.get();
    shambase::DistributedData<NGLink<Tscal>> &flux_rho_dust_face_zm
        = storage.flux_rho_dust_face_zm.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_xp
        = storage.flux_rhov_dust_face_xp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_xm
        = storage.flux_rhov_dust_face_xm.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_yp
        = storage.flux_rhov_dust_face_yp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_ym
        = storage.flux_rhov_dust_face_ym.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_zp
        = storage.flux_rhov_dust_face_zp.get();
    shambase::DistributedData<NGLink<Tvec>> &flux_rhov_dust_face_zm
        = storage.flux_rhov_dust_face_zm.get();

    scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR Flux]", "accumulate fluxes patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        OrientedAMRGraph &oriented_cell_graph
            = shambase::get_check_ref(storage.cell_graph_edge).graph.get(id);

        NGLink<Tscal> &patch_flux_rho_dust_face_xp = flux_rho_dust_face_xp.get(id);
        NGLink<Tscal> &patch_flux_rho_dust_face_xm = flux_rho_dust_face_xm.get(id);
        NGLink<Tscal> &patch_flux_rho_dust_face_yp = flux_rho_dust_face_yp.get(id);
        NGLink<Tscal> &patch_flux_rho_dust_face_ym = flux_rho_dust_face_ym.get(id);
        NGLink<Tscal> &patch_flux_rho_dust_face_zp = flux_rho_dust_face_zp.get(id);
        NGLink<Tscal> &patch_flux_rho_dust_face_zm = flux_rho_dust_face_zm.get(id);

        NGLink<Tvec> &patch_flux_rhov_dust_face_xp = flux_rhov_dust_face_xp.get(id);
        NGLink<Tvec> &patch_flux_rhov_dust_face_xm = flux_rhov_dust_face_xm.get(id);
        NGLink<Tvec> &patch_flux_rhov_dust_face_yp = flux_rhov_dust_face_yp.get(id);
        NGLink<Tvec> &patch_flux_rhov_dust_face_ym = flux_rhov_dust_face_ym.get(id);
        NGLink<Tvec> &patch_flux_rhov_dust_face_zp = flux_rhov_dust_face_zp.get(id);
        NGLink<Tvec> &patch_flux_rhov_dust_face_zm = flux_rhov_dust_face_zm.get(id);

        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_xp
            = patch_flux_rho_dust_face_xp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_xm
            = patch_flux_rho_dust_face_xm.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_yp
            = patch_flux_rho_dust_face_yp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_ym
            = patch_flux_rho_dust_face_ym.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_zp
            = patch_flux_rho_dust_face_zp.link_graph_field;
        sham::DeviceBuffer<Tscal> &buf_flux_rho_dust_face_zm
            = patch_flux_rho_dust_face_zm.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_xp
            = patch_flux_rhov_dust_face_xp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_xm
            = patch_flux_rhov_dust_face_xm.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_yp
            = patch_flux_rhov_dust_face_yp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_ym
            = patch_flux_rhov_dust_face_ym.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_zp
            = patch_flux_rhov_dust_face_zp.link_graph_field;
        sham::DeviceBuffer<Tvec> &buf_flux_rhov_dust_face_zm
            = patch_flux_rhov_dust_face_zm.link_graph_field;

        sham::DeviceBuffer<Tscal> &dt_rho_dust_patch = cfield_dtrho_dust.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_dust_patch = cfield_dtrhov_dust.get_buf_check(id);

        AMRGraph &graph_neigh_xp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        sham::DeviceBuffer<Tscal> &block_cell_sizes
            = shambase::get_check_ref(storage.block_cell_sizes).get_refs().get(id).get().get_buf();
        sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
            = shambase::get_check_ref(storage.cell0block_aabb_lower)
                  .get_refs()
                  .get(id)
                  .get()
                  .get_buf();

        sham::EventList depends_list;
        auto acc_aabb_block_lower   = cell0block_aabb_lower.get_read_access(depends_list);
        auto acc_aabb_cell_size     = block_cell_sizes.get_read_access(depends_list);
        auto acc_dt_rho_dust_patch  = dt_rho_dust_patch.get_write_access(depends_list);
        auto acc_dt_rhov_dust_patch = dt_rhov_dust_patch.get_write_access(depends_list);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        auto flux_rho_dust_face_xp = buf_flux_rho_dust_face_xp.get_read_access(depends_list);
        auto flux_rho_dust_face_xm = buf_flux_rho_dust_face_xm.get_read_access(depends_list);
        auto flux_rho_dust_face_yp = buf_flux_rho_dust_face_yp.get_read_access(depends_list);
        auto flux_rho_dust_face_ym = buf_flux_rho_dust_face_ym.get_read_access(depends_list);
        auto flux_rho_dust_face_zp = buf_flux_rho_dust_face_zp.get_read_access(depends_list);
        auto flux_rho_dust_face_zm = buf_flux_rho_dust_face_zm.get_read_access(depends_list);

        auto flux_rhov_dust_face_xp = buf_flux_rhov_dust_face_xp.get_read_access(depends_list);
        auto flux_rhov_dust_face_xm = buf_flux_rhov_dust_face_xm.get_read_access(depends_list);
        auto flux_rhov_dust_face_yp = buf_flux_rhov_dust_face_yp.get_read_access(depends_list);
        auto flux_rhov_dust_face_ym = buf_flux_rhov_dust_face_ym.get_read_access(depends_list);
        auto flux_rhov_dust_face_zp = buf_flux_rhov_dust_face_zp.get_read_access(depends_list);
        auto flux_rhov_dust_face_zm = buf_flux_rhov_dust_face_zm.get_read_access(depends_list);

        auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
        auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
        auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
        auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
        auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
        auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            u32 ndust          = solver_config.dust_config.ndust;
            auto get_cell_aabb = [=](u32 id) -> shammath::AABB<Tvec> {
                const u32 cell_global_id = (u32) id;

                // block id : this is the block id of the current cell
                const u32 block_id = cell_global_id / AMRBlock::block_size;
                // id of the current cell with respect to it's block
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const Tvec cblock_min  = acc_aabb_block_lower[block_id];
                const Tscal delta_cell = acc_aabb_cell_size[block_id];

                std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

                Tvec aabb_min = cblock_min + offset;
                Tvec aabb_max = aabb_min + delta_cell;

                return {aabb_min, aabb_max};
            };

            // id_a and id_a are both cells ids
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

            shambase::parralel_for(cgh, ndust * cell_count, "accumulate fluxes", [=](u32 ivar_a) {
                // cell id in the global space of index
                const u32 icell_a = ivar_a / ndust;
                // variable id in the cell of icell_a id
                const u32 ndust_off_loc = ivar_a % ndust;

                // block id of the ivar_a
                const u32 block_id = icell_a / AMRBlock::block_size;
                // cell id in the block of id block_id
                const u32 cell_loc_id = icell_a % AMRBlock::block_size;

                Tscal V_i = acc_aabb_cell_size[block_id];
                V_i       = V_i * V_i * V_i;

                Tscal dtrho_dust = 0;
                Tvec dtrhov_dust = {0, 0, 0};

                // iterate trough neighborh table is xp direction
                graph_iter_xp.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_xp[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_xp[link_id * ndust + ndust_off_loc] * S_ij;
                });

                // iterate trough neighborh table is yp direction
                graph_iter_yp.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_yp[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_yp[link_id * ndust + ndust_off_loc] * S_ij;
                });

                // iterate trough neighborh table is zp direction
                graph_iter_zp.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_zp[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_zp[link_id * ndust + ndust_off_loc] * S_ij;
                });

                // iterate trough neighborh table is xm direction
                graph_iter_xm.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_xm[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_xm[link_id * ndust + ndust_off_loc] * S_ij;
                });

                // iterate trough neighborh table is ym direction
                graph_iter_ym.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_ym[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_ym[link_id * ndust + ndust_off_loc] * S_ij;
                });

                // iterate trough neighborh table is zm direction
                graph_iter_zm.for_each_object_link_id(icell_a, [&](u32 icell_b, u32 link_id) {
                    Tscal S_ij = get_face_surface(icell_a, icell_b);
                    dtrho_dust -= flux_rho_dust_face_zm[link_id * ndust + ndust_off_loc] * S_ij;
                    dtrhov_dust -= flux_rhov_dust_face_zm[link_id * ndust + ndust_off_loc] * S_ij;
                });

                dtrho_dust /= V_i;
                dtrhov_dust /= V_i;

                acc_dt_rho_dust_patch[icell_a * ndust + ndust_off_loc]  = dtrho_dust;
                acc_dt_rhov_dust_patch[icell_a * ndust + ndust_off_loc] = dtrhov_dust;
            });
        });

        cell0block_aabb_lower.complete_event_state(e);
        block_cell_sizes.complete_event_state(e);
        dt_rho_dust_patch.complete_event_state(e);
        dt_rhov_dust_patch.complete_event_state(e);

        buf_flux_rho_dust_face_xp.complete_event_state(e);
        buf_flux_rho_dust_face_xm.complete_event_state(e);
        buf_flux_rho_dust_face_yp.complete_event_state(e);
        buf_flux_rho_dust_face_ym.complete_event_state(e);
        buf_flux_rho_dust_face_zp.complete_event_state(e);
        buf_flux_rho_dust_face_zm.complete_event_state(e);

        buf_flux_rhov_dust_face_xp.complete_event_state(e);
        buf_flux_rhov_dust_face_xm.complete_event_state(e);
        buf_flux_rhov_dust_face_yp.complete_event_state(e);
        buf_flux_rhov_dust_face_ym.complete_event_state(e);
        buf_flux_rhov_dust_face_zp.complete_event_state(e);
        buf_flux_rhov_dust_face_zm.complete_event_state(e);

        graph_neigh_xp.complete_event_state(e);
        graph_neigh_xm.complete_event_state(e);
        graph_neigh_yp.complete_event_state(e);
        graph_neigh_ym.complete_event_state(e);
        graph_neigh_zp.complete_event_state(e);
        graph_neigh_zm.complete_event_state(e);
    });

    storage.dtrho_dust.set(std::move(cfield_dtrho_dust));
    storage.dtrhov_dust.set(std::move(cfield_dtrhov_dust));
}

template class shammodels::basegodunov::modules::ComputeTimeDerivative<f64_3, i64_3>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGUtilities.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGUtilities.hpp"
#include "shammodels/ramses/modules/ComputeRhoMean.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

namespace {
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
        const Tvec delta_cell,
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

        T inv_dx_sqr = 1.0 / (delta_cell.x() * delta_cell.x());
        T inv_dy_sqr = 1.0 / (delta_cell.y() * delta_cell.y());
        T inv_dz_sqr = 1.0 / (delta_cell.z() * delta_cell.z());

        T laplace_x = inv_dx_sqr * (W_xm - 2. * W_i + W_xp);
        T laplace_y = inv_dy_sqr * (W_ym - 2. * W_i + W_yp);
        T laplace_z = inv_dz_sqr * (W_zm - 2. * W_i + W_zp);

        return (laplace_x + laplace_y + laplace_z);
    }

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    auto shammodels::basegodunov::modules::CGUtilities<Tvec, TgridVec>::init_step() -> void {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        using namespace shamrock;
        using namespace shammath;
        using MergedPDat = shamrock::MergedPatchData;
        shamrock::SchedulerUtility utility(scheduler());

        ComputeField<Tscal> residual
            = utility.make_compute_field<Tscal>("phi_res", AMRBlock::block_size); // residuals

        ComputeField<Tscal> p = utility.make_compute_field<Tscal>(
            "phi_p", AMRBlock::block_size); // vector p (searching direction)
        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
        u32 iphi_ghost                                 = ghost_layout.get_field_idx<Tscal>("phi");
        u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");

        Tscal fourPiG = solver_config.fourPiG();
        ComputeRhoMean comp_rho_mean(context, solver_config, storage);
        Tscal rho_mean = comp_rho_mean.compute_rho_mean();

        storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
            sham::DeviceBuffer<Tscal> &buf_phi = mpdat.pdat.get_field_buf_ref<Tscal>(iphi_ghost);

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

            sham::EventList depends_list;
            auto acc_block_min = buf_block_min.get_read_access(depends_list);
            auto acc_block_max = buf_block_max.get_read_access(depends_list);
            auto acc_rho       = buf_rho.get_read_access(depends_list);
            auto acc_phi       = buf_phi.get_write_access(depends_list);
            auto acc_residual  = residual.get_buf(id).get_write_access(depends_list);
            auto acc_p         = p.get_buf(id).get_write_access(depends_list);

            auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
            auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
            auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
            auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
            auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
            auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                u32 cell_count       = (mpdat.total_elements) * AMRBlock::block_size;
                Tscal one_over_Nside = 1. / AMRBlock::Nside;
                Tscal dxfact         = solver_config.grid_coord_to_pos_fact;

                shambase::parralel_for(cgh, cell_count, "self_gravity::init_step", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;

                    const u32 block_id    = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;
                    TgridVec lower        = acc_block_min[block_id];
                    TgridVec upper        = acc_block_max[block_id];
                    Tvec lower_flt        = lower.template convert<Tscal>() * dxfact;
                    Tvec upper_flt        = upper.template convert<Tscal>() * dxfact;
                    Tvec delta_cell       = (upper_flt - lower_flt) * one_over_Nside;

                    auto Aphi_id = get_gemv_id<Tscal, Tvec>(
                        cell_global_id,
                        delta_cell,
                        graph_iter_xp,
                        graph_iter_xm,
                        graph_iter_yp,
                        graph_iter_ym,
                        graph_iter_zp,
                        graph_iter_zm,
                        [=](u32 id) {
                            return acc_phi[id];
                        });
                    // Here we suppose periodic BC for the right hand side
                    auto res = fourPiG * (acc_rho[cell_global_id] - rho_mean)
                               - Aphi_id; // r_k = 4*PI*G*(\rho - \Bar{rho}) - A*x_k
                    acc_residual[cell_global_id] = res;
                    // init the seaching direction vector
                    acc_p[cell_global_id] = res; // p_0 = r_0
                });
            });
            buf_block_min.complete_event_state(e);
            buf_block_max.complete_event_state(e);
            buf_rho.complete_event_state(e);
            buf_phi.complete_event_state(e);
            residual.get_buf(id).complete_event_state(e);
            p.get_buf(id).complete_event_state(e);

            graph_neigh_xp.complete_event_state(e);
            graph_neigh_xm.complete_event_state(e);
            graph_neigh_yp.complete_event_state(e);
            graph_neigh_ym.complete_event_state(e);
            graph_neigh_zp.complete_event_state(e);
            graph_neigh_zm.complete_event_state(e);
        });
        storage.phi_p.set(std::move(p));
        storage.phi_res.set(std::move(residual));
    }

    template<class Tvec, class TgridVec>
    auto
    shammodels::basegodunov::modules::CGUtilities<Tvec, TgridVec>::compute_ddot_res() -> Tscal {
        StackEntry stack_loc{};
        shamrock::ComputeField<Tscal> &cfield_phi_res = storage.phi_res.get();
        Tscal rank_ddot = cfield_phi_res.compute_rank_dot_sum(); // dot product per patch
        Tscal tot_ddot
            = shamalgs::collective::allreduce_sum(rank_ddot); // total dot procduct over the grid
        return tot_ddot;
    }

    template<class Tvec, class TgridVec>
    auto shammodels::basegodunov::modules::CGUtilities<Tvec, TgridVec>::compute_Ap() -> void {
        StackEntry stack_loc{};
        using namespace shamrock::patch;
        using namespace shamrock;
        using namespace shammath;
        using MergedPDat = shamrock::MergedPatchData;

        SchedulerUtility utility(scheduler());
        ComputeField<Tscal> Ap = utility.make_compute_field<Tscal>("phi_Ap", AMRBlock::block_size);

        storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

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

            sham::DeviceBuffer<Tscal> &buf_phi_p = storage.phi_p.get().get_buf(id);

            sham::EventList depends_list;
            auto acc_block_min = buf_block_min.get_read_access(depends_list);
            auto acc_block_max = buf_block_max.get_read_access(depends_list);
            auto acc_Ap        = Ap.get_buf(id).get_write_access(depends_list);
            auto acc_p         = buf_phi_p.get_read_access(depends_list);

            auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
            auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
            auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
            auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
            auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
            auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                u32 cell_count       = (mpdat.total_elements) * AMRBlock::block_size;
                Tscal one_over_Nside = 1. / AMRBlock::Nside;
                Tscal dxfact         = solver_config.grid_coord_to_pos_fact;
                shambase::parralel_for(cgh, cell_count, "self_gravity::Ap", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;

                    const u32 block_id    = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;
                    TgridVec lower        = acc_block_min[block_id];
                    TgridVec upper        = acc_block_max[block_id];
                    Tvec lower_flt        = lower.template convert<Tscal>() * dxfact;
                    Tvec upper_flt        = upper.template convert<Tscal>() * dxfact;
                    Tvec delta_cell       = (upper_flt - lower_flt) * one_over_Nside;

                    auto Ap_id = get_gemv_id<Tscal, Tvec>(
                        cell_global_id,
                        delta_cell,
                        graph_iter_xp,
                        graph_iter_xm,
                        graph_iter_yp,
                        graph_iter_ym,
                        graph_iter_zp,
                        graph_iter_zm,
                        [=](u32 id) {
                            return acc_p[id];
                        });
                    acc_Ap[cell_global_id] = Ap_id;
                });
            });
            buf_block_min.complete_event_state(e);
            buf_block_max.complete_event_state(e);
            Ap.get_buf(id).complete_event_state(e);
            buf_phi_p.complete_event_state(e);

            graph_neigh_xp.complete_event_state(e);
            graph_neigh_xm.complete_event_state(e);
            graph_neigh_yp.complete_event_state(e);
            graph_neigh_ym.complete_event_state(e);
            graph_neigh_zp.complete_event_state(e);
            graph_neigh_zm.complete_event_state(e);
        });
        storage.phi_Ap.set(std::move(Ap));
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::CGUtilities<f64_3, i64_3>;

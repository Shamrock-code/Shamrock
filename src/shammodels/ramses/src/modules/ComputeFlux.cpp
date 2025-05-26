// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeFlux.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/ComputeFlux.hpp"
#include "shammodels/ramses/modules/ComputeFluxUtilities.hpp"
#include <array>

template<class T>
using NGLink = shammodels::basegodunov::modules::NeighGraphLinkField<T>;

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeFlux<Tvec, TgridVec>::compute_flux() {

    StackEntry stack_loc{};

    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_xp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_xm;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_yp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_ym;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_zp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_face_zm;

    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_xp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_xm;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_yp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_ym;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_zp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_face_zm;

    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_xp;
    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_xm;
    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_yp;
    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_ym;
    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_zp;
    shambase::DistributedData<NGLink<Tscal>> flux_rhoe_face_zm;

    Tscal gamma = solver_config.eos_gamma;

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            NGLink<std::array<Tscal, 2>> &rho_face_xp = storage.rho_face_xp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_face_xm = storage.rho_face_xm.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_face_yp = storage.rho_face_yp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_face_ym = storage.rho_face_ym.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_face_zp = storage.rho_face_zp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_face_zm = storage.rho_face_zm.get().get(id);

            NGLink<std::array<Tvec, 2>> &vel_face_xp = storage.vel_face_xp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_face_xm = storage.vel_face_xm.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_face_yp = storage.vel_face_yp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_face_ym = storage.vel_face_ym.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_face_zp = storage.vel_face_zp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_face_zm = storage.vel_face_zm.get().get(id);

            NGLink<std::array<Tscal, 2>> &press_face_xp = storage.press_face_xp.get().get(id);
            NGLink<std::array<Tscal, 2>> &press_face_xm = storage.press_face_xm.get().get(id);
            NGLink<std::array<Tscal, 2>> &press_face_yp = storage.press_face_yp.get().get(id);
            NGLink<std::array<Tscal, 2>> &press_face_ym = storage.press_face_ym.get().get(id);
            NGLink<std::array<Tscal, 2>> &press_face_zp = storage.press_face_zp.get().get(id);
            NGLink<std::array<Tscal, 2>> &press_face_zm = storage.press_face_zm.get().get(id);

            const u32 ixp = oriented_cell_graph.xp;
            const u32 ixm = oriented_cell_graph.xm;
            const u32 iyp = oriented_cell_graph.yp;
            const u32 iym = oriented_cell_graph.ym;
            const u32 izp = oriented_cell_graph.zp;
            const u32 izm = oriented_cell_graph.zm;

            NGLink<Tscal> buf_flux_rho_face_xp{*oriented_cell_graph.graph_links[ixp]};
            NGLink<Tscal> buf_flux_rho_face_xm{*oriented_cell_graph.graph_links[ixm]};
            NGLink<Tscal> buf_flux_rho_face_yp{*oriented_cell_graph.graph_links[iyp]};
            NGLink<Tscal> buf_flux_rho_face_ym{*oriented_cell_graph.graph_links[iym]};
            NGLink<Tscal> buf_flux_rho_face_zp{*oriented_cell_graph.graph_links[izp]};
            NGLink<Tscal> buf_flux_rho_face_zm{*oriented_cell_graph.graph_links[izm]};

            NGLink<Tvec> buf_flux_rhov_face_xp{*oriented_cell_graph.graph_links[ixp]};
            NGLink<Tvec> buf_flux_rhov_face_xm{*oriented_cell_graph.graph_links[ixm]};
            NGLink<Tvec> buf_flux_rhov_face_yp{*oriented_cell_graph.graph_links[iyp]};
            NGLink<Tvec> buf_flux_rhov_face_ym{*oriented_cell_graph.graph_links[iym]};
            NGLink<Tvec> buf_flux_rhov_face_zp{*oriented_cell_graph.graph_links[izp]};
            NGLink<Tvec> buf_flux_rhov_face_zm{*oriented_cell_graph.graph_links[izm]};

            NGLink<Tscal> buf_flux_rhoe_face_xp{*oriented_cell_graph.graph_links[ixp]};
            NGLink<Tscal> buf_flux_rhoe_face_xm{*oriented_cell_graph.graph_links[ixm]};
            NGLink<Tscal> buf_flux_rhoe_face_yp{*oriented_cell_graph.graph_links[iyp]};
            NGLink<Tscal> buf_flux_rhoe_face_ym{*oriented_cell_graph.graph_links[iym]};
            NGLink<Tscal> buf_flux_rhoe_face_zp{*oriented_cell_graph.graph_links[izp]};
            NGLink<Tscal> buf_flux_rhoe_face_zm{*oriented_cell_graph.graph_links[izm]};

            if (solver_config.riemman_config == Rusanov) {
                constexpr RiemmanSolverMode mode = Rusanov;
                logger::debug_ln("[AMR Flux]", "compute rusanov xp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                    q,
                    rho_face_xp.link_count,
                    rho_face_xp.link_graph_field,
                    vel_face_xp.link_graph_field,
                    press_face_xp.link_graph_field,
                    buf_flux_rho_face_xp.link_graph_field,
                    buf_flux_rhov_face_xp.link_graph_field,
                    buf_flux_rhoe_face_xp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute rusanov yp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                    q,
                    rho_face_yp.link_count,
                    rho_face_yp.link_graph_field,
                    vel_face_yp.link_graph_field,
                    press_face_yp.link_graph_field,
                    buf_flux_rho_face_yp.link_graph_field,
                    buf_flux_rhov_face_yp.link_graph_field,
                    buf_flux_rhoe_face_yp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute rusanov zp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                    q,
                    rho_face_zp.link_count,
                    rho_face_zp.link_graph_field,
                    vel_face_zp.link_graph_field,
                    press_face_zp.link_graph_field,
                    buf_flux_rho_face_zp.link_graph_field,
                    buf_flux_rhov_face_zp.link_graph_field,
                    buf_flux_rhoe_face_zp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute rusanov xm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                    q,
                    rho_face_xm.link_count,
                    rho_face_xm.link_graph_field,
                    vel_face_xm.link_graph_field,
                    press_face_xm.link_graph_field,
                    buf_flux_rho_face_xm.link_graph_field,
                    buf_flux_rhov_face_xm.link_graph_field,
                    buf_flux_rhoe_face_xm.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute rusanov ym patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                    q,
                    rho_face_ym.link_count,
                    rho_face_ym.link_graph_field,
                    vel_face_ym.link_graph_field,
                    press_face_ym.link_graph_field,
                    buf_flux_rho_face_ym.link_graph_field,
                    buf_flux_rhov_face_ym.link_graph_field,
                    buf_flux_rhoe_face_ym.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute rusanov zm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                    q,
                    rho_face_zm.link_count,
                    rho_face_zm.link_graph_field,
                    vel_face_zm.link_graph_field,
                    press_face_zm.link_graph_field,
                    buf_flux_rho_face_zm.link_graph_field,
                    buf_flux_rhov_face_zm.link_graph_field,
                    buf_flux_rhoe_face_zm.link_graph_field,
                    gamma);
            } else if (solver_config.riemman_config == HLL) {
                constexpr RiemmanSolverMode mode = HLL;
                logger::debug_ln("[AMR Flux]", "compute HLL xp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                    q,
                    rho_face_xp.link_count,
                    rho_face_xp.link_graph_field,
                    vel_face_xp.link_graph_field,
                    press_face_xp.link_graph_field,
                    buf_flux_rho_face_xp.link_graph_field,
                    buf_flux_rhov_face_xp.link_graph_field,
                    buf_flux_rhoe_face_xp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLL yp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                    q,
                    rho_face_yp.link_count,
                    rho_face_yp.link_graph_field,
                    vel_face_yp.link_graph_field,
                    press_face_yp.link_graph_field,
                    buf_flux_rho_face_yp.link_graph_field,
                    buf_flux_rhov_face_yp.link_graph_field,
                    buf_flux_rhoe_face_yp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLL zp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                    q,
                    rho_face_zp.link_count,
                    rho_face_zp.link_graph_field,
                    vel_face_zp.link_graph_field,
                    press_face_zp.link_graph_field,
                    buf_flux_rho_face_zp.link_graph_field,
                    buf_flux_rhov_face_zp.link_graph_field,
                    buf_flux_rhoe_face_zp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLL xm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                    q,
                    rho_face_xm.link_count,
                    rho_face_xm.link_graph_field,
                    vel_face_xm.link_graph_field,
                    press_face_xm.link_graph_field,
                    buf_flux_rho_face_xm.link_graph_field,
                    buf_flux_rhov_face_xm.link_graph_field,
                    buf_flux_rhoe_face_xm.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLL ym patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                    q,
                    rho_face_ym.link_count,
                    rho_face_ym.link_graph_field,
                    vel_face_ym.link_graph_field,
                    press_face_ym.link_graph_field,
                    buf_flux_rho_face_ym.link_graph_field,
                    buf_flux_rhov_face_ym.link_graph_field,
                    buf_flux_rhoe_face_ym.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLL zm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                    q,
                    rho_face_zm.link_count,
                    rho_face_zm.link_graph_field,
                    vel_face_zm.link_graph_field,
                    press_face_zm.link_graph_field,
                    buf_flux_rho_face_zm.link_graph_field,
                    buf_flux_rhov_face_zm.link_graph_field,
                    buf_flux_rhoe_face_zm.link_graph_field,
                    gamma);
            } else if (solver_config.riemman_config == HLLC) {
                constexpr RiemmanSolverMode mode = HLLC;
                logger::debug_ln("[AMR Flux]", "compute HLLC xp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                    q,
                    rho_face_xp.link_count,
                    rho_face_xp.link_graph_field,
                    vel_face_xp.link_graph_field,
                    press_face_xp.link_graph_field,
                    buf_flux_rho_face_xp.link_graph_field,
                    buf_flux_rhov_face_xp.link_graph_field,
                    buf_flux_rhoe_face_xp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLLC yp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                    q,
                    rho_face_yp.link_count,
                    rho_face_yp.link_graph_field,
                    vel_face_yp.link_graph_field,
                    press_face_yp.link_graph_field,
                    buf_flux_rho_face_yp.link_graph_field,
                    buf_flux_rhov_face_yp.link_graph_field,
                    buf_flux_rhoe_face_yp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLLC zp patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                    q,
                    rho_face_zp.link_count,
                    rho_face_zp.link_graph_field,
                    vel_face_zp.link_graph_field,
                    press_face_zp.link_graph_field,
                    buf_flux_rho_face_zp.link_graph_field,
                    buf_flux_rhov_face_zp.link_graph_field,
                    buf_flux_rhoe_face_zp.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLLC xm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                    q,
                    rho_face_xm.link_count,
                    rho_face_xm.link_graph_field,
                    vel_face_xm.link_graph_field,
                    press_face_xm.link_graph_field,
                    buf_flux_rho_face_xm.link_graph_field,
                    buf_flux_rhov_face_xm.link_graph_field,
                    buf_flux_rhoe_face_xm.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLLC ym patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                    q,
                    rho_face_ym.link_count,
                    rho_face_ym.link_graph_field,
                    vel_face_ym.link_graph_field,
                    press_face_ym.link_graph_field,
                    buf_flux_rho_face_ym.link_graph_field,
                    buf_flux_rhov_face_ym.link_graph_field,
                    buf_flux_rhoe_face_ym.link_graph_field,
                    gamma);
                logger::debug_ln("[AMR Flux]", "compute HLLC zm patch", id);
                compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                    q,
                    rho_face_zm.link_count,
                    rho_face_zm.link_graph_field,
                    vel_face_zm.link_graph_field,
                    press_face_zm.link_graph_field,
                    buf_flux_rho_face_zm.link_graph_field,
                    buf_flux_rhov_face_zm.link_graph_field,
                    buf_flux_rhoe_face_zm.link_graph_field,
                    gamma);
            }

            flux_rho_face_xp.add_obj(id, std::move(buf_flux_rho_face_xp));
            flux_rho_face_xm.add_obj(id, std::move(buf_flux_rho_face_xm));
            flux_rho_face_yp.add_obj(id, std::move(buf_flux_rho_face_yp));
            flux_rho_face_ym.add_obj(id, std::move(buf_flux_rho_face_ym));
            flux_rho_face_zp.add_obj(id, std::move(buf_flux_rho_face_zp));
            flux_rho_face_zm.add_obj(id, std::move(buf_flux_rho_face_zm));

            flux_rhov_face_xp.add_obj(id, std::move(buf_flux_rhov_face_xp));
            flux_rhov_face_xm.add_obj(id, std::move(buf_flux_rhov_face_xm));
            flux_rhov_face_yp.add_obj(id, std::move(buf_flux_rhov_face_yp));
            flux_rhov_face_ym.add_obj(id, std::move(buf_flux_rhov_face_ym));
            flux_rhov_face_zp.add_obj(id, std::move(buf_flux_rhov_face_zp));
            flux_rhov_face_zm.add_obj(id, std::move(buf_flux_rhov_face_zm));

            flux_rhoe_face_xp.add_obj(id, std::move(buf_flux_rhoe_face_xp));
            flux_rhoe_face_xm.add_obj(id, std::move(buf_flux_rhoe_face_xm));
            flux_rhoe_face_yp.add_obj(id, std::move(buf_flux_rhoe_face_yp));
            flux_rhoe_face_ym.add_obj(id, std::move(buf_flux_rhoe_face_ym));
            flux_rhoe_face_zp.add_obj(id, std::move(buf_flux_rhoe_face_zp));
            flux_rhoe_face_zm.add_obj(id, std::move(buf_flux_rhoe_face_zm));
        });

    storage.flux_rho_face_xp.set(std::move(flux_rho_face_xp));
    storage.flux_rho_face_xm.set(std::move(flux_rho_face_xm));
    storage.flux_rho_face_yp.set(std::move(flux_rho_face_yp));
    storage.flux_rho_face_ym.set(std::move(flux_rho_face_ym));
    storage.flux_rho_face_zp.set(std::move(flux_rho_face_zp));
    storage.flux_rho_face_zm.set(std::move(flux_rho_face_zm));
    storage.flux_rhov_face_xp.set(std::move(flux_rhov_face_xp));
    storage.flux_rhov_face_xm.set(std::move(flux_rhov_face_xm));
    storage.flux_rhov_face_yp.set(std::move(flux_rhov_face_yp));
    storage.flux_rhov_face_ym.set(std::move(flux_rhov_face_ym));
    storage.flux_rhov_face_zp.set(std::move(flux_rhov_face_zp));
    storage.flux_rhov_face_zm.set(std::move(flux_rhov_face_zm));
    storage.flux_rhoe_face_xp.set(std::move(flux_rhoe_face_xp));
    storage.flux_rhoe_face_xm.set(std::move(flux_rhoe_face_xm));
    storage.flux_rhoe_face_yp.set(std::move(flux_rhoe_face_yp));
    storage.flux_rhoe_face_ym.set(std::move(flux_rhoe_face_ym));
    storage.flux_rhoe_face_zp.set(std::move(flux_rhoe_face_zp));
    storage.flux_rhoe_face_zm.set(std::move(flux_rhoe_face_zm));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeFlux<Tvec, TgridVec>::compute_flux_dust() {

    StackEntry stack_loc{};

    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_xp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_xm;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_yp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_ym;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_zp;
    shambase::DistributedData<NGLink<Tscal>> flux_rho_dust_face_zm;

    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_xp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_xm;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_yp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_ym;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_zp;
    shambase::DistributedData<NGLink<Tvec>> flux_rhov_dust_face_zm;

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            NGLink<std::array<Tscal, 2>> &rho_dust_face_xp = storage.rho_dust_face_xp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_dust_face_xm = storage.rho_dust_face_xm.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_dust_face_yp = storage.rho_dust_face_yp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_dust_face_ym = storage.rho_dust_face_ym.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_dust_face_zp = storage.rho_dust_face_zp.get().get(id);
            NGLink<std::array<Tscal, 2>> &rho_dust_face_zm = storage.rho_dust_face_zm.get().get(id);

            NGLink<std::array<Tvec, 2>> &vel_dust_face_xp = storage.vel_dust_face_xp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_dust_face_xm = storage.vel_dust_face_xm.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_dust_face_yp = storage.vel_dust_face_yp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_dust_face_ym = storage.vel_dust_face_ym.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_dust_face_zp = storage.vel_dust_face_zp.get().get(id);
            NGLink<std::array<Tvec, 2>> &vel_dust_face_zm = storage.vel_dust_face_zm.get().get(id);

            const u32 ixp = oriented_cell_graph.xp;
            const u32 ixm = oriented_cell_graph.xm;
            const u32 iyp = oriented_cell_graph.yp;
            const u32 iym = oriented_cell_graph.ym;
            const u32 izp = oriented_cell_graph.zp;
            const u32 izm = oriented_cell_graph.zm;

            auto ndust = solver_config.dust_config.ndust;

            NGLink<Tscal> buf_flux_rho_dust_face_xp{*oriented_cell_graph.graph_links[ixp], ndust};
            NGLink<Tscal> buf_flux_rho_dust_face_xm{*oriented_cell_graph.graph_links[ixm], ndust};
            NGLink<Tscal> buf_flux_rho_dust_face_yp{*oriented_cell_graph.graph_links[iyp], ndust};
            NGLink<Tscal> buf_flux_rho_dust_face_ym{*oriented_cell_graph.graph_links[iym], ndust};
            NGLink<Tscal> buf_flux_rho_dust_face_zp{*oriented_cell_graph.graph_links[izp], ndust};
            NGLink<Tscal> buf_flux_rho_dust_face_zm{*oriented_cell_graph.graph_links[izm], ndust};

            NGLink<Tvec> buf_flux_rhov_dust_face_xp{*oriented_cell_graph.graph_links[ixp], ndust};
            NGLink<Tvec> buf_flux_rhov_dust_face_xm{*oriented_cell_graph.graph_links[ixm], ndust};
            NGLink<Tvec> buf_flux_rhov_dust_face_yp{*oriented_cell_graph.graph_links[iyp], ndust};
            NGLink<Tvec> buf_flux_rhov_dust_face_ym{*oriented_cell_graph.graph_links[iym], ndust};
            NGLink<Tvec> buf_flux_rhov_dust_face_zp{*oriented_cell_graph.graph_links[izp], ndust};
            NGLink<Tvec> buf_flux_rhov_dust_face_zm{*oriented_cell_graph.graph_links[izm], ndust};

            u32 _ndust = solver_config.dust_config.ndust;
            if (solver_config.dust_config.dust_riemann_config == DHLL) {
                constexpr DustRiemannSolverMode mode = DHLL;
                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll xp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                    q,
                    rho_dust_face_xp.link_count,
                    rho_dust_face_xp.link_graph_field,
                    vel_dust_face_xp.link_graph_field,
                    buf_flux_rho_dust_face_xp.link_graph_field,
                    buf_flux_rhov_dust_face_xp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll yp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                    q,
                    rho_dust_face_yp.link_count,
                    rho_dust_face_yp.link_graph_field,
                    vel_dust_face_yp.link_graph_field,
                    buf_flux_rho_dust_face_yp.link_graph_field,
                    buf_flux_rhov_dust_face_yp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll zp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                    q,
                    rho_dust_face_zp.link_count,
                    rho_dust_face_zp.link_graph_field,
                    vel_dust_face_zp.link_graph_field,
                    buf_flux_rho_dust_face_zp.link_graph_field,
                    buf_flux_rhov_dust_face_zp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll xm patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                    q,
                    rho_dust_face_xm.link_count,
                    rho_dust_face_xm.link_graph_field,
                    vel_dust_face_xm.link_graph_field,
                    buf_flux_rho_dust_face_xm.link_graph_field,
                    buf_flux_rhov_dust_face_xm.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll ym patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                    q,
                    rho_dust_face_ym.link_count,
                    rho_dust_face_ym.link_graph_field,
                    vel_dust_face_ym.link_graph_field,
                    buf_flux_rho_dust_face_ym.link_graph_field,
                    buf_flux_rhov_dust_face_ym.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll zm patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                    q,
                    rho_dust_face_zm.link_count,
                    rho_dust_face_zm.link_graph_field,
                    vel_dust_face_zm.link_graph_field,
                    buf_flux_rho_dust_face_zm.link_graph_field,
                    buf_flux_rhov_dust_face_zm.link_graph_field,
                    _ndust);
            } else if (solver_config.dust_config.dust_riemann_config == HB) {

                constexpr DustRiemannSolverMode mode = HB;
                logger::debug_ln("[AMR Flux]", "compute dust huang-bai xp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                    q,
                    rho_dust_face_xp.link_count,
                    rho_dust_face_xp.link_graph_field,
                    vel_dust_face_xp.link_graph_field,
                    buf_flux_rho_dust_face_xp.link_graph_field,
                    buf_flux_rhov_dust_face_xp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust huang-bai yp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                    q,
                    rho_dust_face_yp.link_count,
                    rho_dust_face_yp.link_graph_field,
                    vel_dust_face_yp.link_graph_field,
                    buf_flux_rho_dust_face_yp.link_graph_field,
                    buf_flux_rhov_dust_face_yp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust huang-bai zp patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                    q,
                    rho_dust_face_zp.link_count,
                    rho_dust_face_zp.link_graph_field,
                    vel_dust_face_zp.link_graph_field,
                    buf_flux_rho_dust_face_zp.link_graph_field,
                    buf_flux_rhov_dust_face_zp.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust huang-bai xm patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                    q,
                    rho_dust_face_xm.link_count,
                    rho_dust_face_xm.link_graph_field,
                    vel_dust_face_xm.link_graph_field,
                    buf_flux_rho_dust_face_xm.link_graph_field,
                    buf_flux_rhov_dust_face_xm.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust huang-bai ym patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                    q,
                    rho_dust_face_ym.link_count,
                    rho_dust_face_ym.link_graph_field,
                    vel_dust_face_ym.link_graph_field,
                    buf_flux_rho_dust_face_ym.link_graph_field,
                    buf_flux_rhov_dust_face_ym.link_graph_field,
                    _ndust);

                logger::debug_ln("[AMR Flux]", "compute dust huang-bai zm patch", id);
                dust_compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                    q,
                    rho_dust_face_zm.link_count,
                    rho_dust_face_zm.link_graph_field,
                    vel_dust_face_zm.link_graph_field,
                    buf_flux_rho_dust_face_zm.link_graph_field,
                    buf_flux_rhov_dust_face_zm.link_graph_field,
                    _ndust);
            }

            flux_rho_dust_face_xp.add_obj(id, std::move(buf_flux_rho_dust_face_xp));
            flux_rho_dust_face_xm.add_obj(id, std::move(buf_flux_rho_dust_face_xm));
            flux_rho_dust_face_yp.add_obj(id, std::move(buf_flux_rho_dust_face_yp));
            flux_rho_dust_face_ym.add_obj(id, std::move(buf_flux_rho_dust_face_ym));
            flux_rho_dust_face_zp.add_obj(id, std::move(buf_flux_rho_dust_face_zp));
            flux_rho_dust_face_zm.add_obj(id, std::move(buf_flux_rho_dust_face_zm));

            flux_rhov_dust_face_xp.add_obj(id, std::move(buf_flux_rhov_dust_face_xp));
            flux_rhov_dust_face_xm.add_obj(id, std::move(buf_flux_rhov_dust_face_xm));
            flux_rhov_dust_face_yp.add_obj(id, std::move(buf_flux_rhov_dust_face_yp));
            flux_rhov_dust_face_ym.add_obj(id, std::move(buf_flux_rhov_dust_face_ym));
            flux_rhov_dust_face_zp.add_obj(id, std::move(buf_flux_rhov_dust_face_zp));
            flux_rhov_dust_face_zm.add_obj(id, std::move(buf_flux_rhov_dust_face_zm));
        });

    storage.flux_rho_dust_face_xp.set(std::move(flux_rho_dust_face_xp));
    storage.flux_rho_dust_face_xm.set(std::move(flux_rho_dust_face_xm));
    storage.flux_rho_dust_face_yp.set(std::move(flux_rho_dust_face_yp));
    storage.flux_rho_dust_face_ym.set(std::move(flux_rho_dust_face_ym));
    storage.flux_rho_dust_face_zp.set(std::move(flux_rho_dust_face_zp));
    storage.flux_rho_dust_face_zm.set(std::move(flux_rho_dust_face_zm));
    storage.flux_rhov_dust_face_xp.set(std::move(flux_rhov_dust_face_xp));
    storage.flux_rhov_dust_face_xm.set(std::move(flux_rhov_dust_face_xm));
    storage.flux_rhov_dust_face_yp.set(std::move(flux_rhov_dust_face_yp));
    storage.flux_rhov_dust_face_ym.set(std::move(flux_rhov_dust_face_ym));
    storage.flux_rhov_dust_face_zp.set(std::move(flux_rhov_dust_face_zp));
    storage.flux_rhov_dust_face_zm.set(std::move(flux_rhov_dust_face_zm));
}

template class shammodels::basegodunov::modules::ComputeFlux<f64_3, i64_3>;

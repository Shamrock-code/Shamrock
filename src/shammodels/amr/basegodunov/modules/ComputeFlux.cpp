// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeFlux.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/ComputeFlux.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeDustFluxUtilities.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeFluxUtilities.hpp"


using RiemmanSolverMode = shammodels::basegodunov::RiemmanSolverMode;
using Direction             = shammodels::basegodunov::modules::Direction;


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


    Tscal gamma = solver_config.eos_gamma;

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        NGLink<std::array<Tscal, 2>> &rho_face_xp = storage.rho_face_xp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_face_xm = storage.rho_face_xm.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_face_yp = storage.rho_face_yp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_face_ym = storage.rho_face_ym.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_face_zp = storage.rho_face_zp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_face_zm = storage.rho_face_zm.get().get(id);

        NGLink<std::array<Tvec, 2>> &rhov_face_xp = storage.rhov_face_xp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_face_xm = storage.rhov_face_xm.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_face_yp = storage.rhov_face_yp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_face_ym = storage.rhov_face_ym.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_face_zp = storage.rhov_face_zp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_face_zm = storage.rhov_face_zm.get().get(id);

        NGLink<std::array<Tscal, 2>> &rhoe_face_xp = storage.rhoe_face_xp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rhoe_face_xm = storage.rhoe_face_xm.get().get(id);
        NGLink<std::array<Tscal, 2>> &rhoe_face_yp = storage.rhoe_face_yp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rhoe_face_ym = storage.rhoe_face_ym.get().get(id);
        NGLink<std::array<Tscal, 2>> &rhoe_face_zp = storage.rhoe_face_zp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rhoe_face_zm = storage.rhoe_face_zm.get().get(id);


        NGLink<std::array<Tscal, 2>> &rho_dust_face_xp = storage.rho_dust_face_xp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_dust_face_xm = storage.rho_dust_face_xm.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_dust_face_yp = storage.rho_dust_face_yp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_dust_face_ym = storage.rho_dust_face_ym.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_dust_face_zp = storage.rho_dust_face_zp.get().get(id);
        NGLink<std::array<Tscal, 2>> &rho_dust_face_zm = storage.rho_dust_face_zm.get().get(id);

        NGLink<std::array<Tvec, 2>> &rhov_dust_face_xp = storage.rhov_dust_face_xp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_dust_face_xm = storage.rhov_dust_face_xm.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_dust_face_yp = storage.rhov_dust_face_yp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_dust_face_ym = storage.rhov_dust_face_ym.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_dust_face_zp = storage.rhov_dust_face_zp.get().get(id);
        NGLink<std::array<Tvec, 2>> &rhov_dust_face_zm = storage.rhov_dust_face_zm.get().get(id);


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


        NGLink<Tscal> buf_flux_rho_dust_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<Tscal> buf_flux_rho_dust_face_xm{*oriented_cell_graph.graph_links[ixm]};
        NGLink<Tscal> buf_flux_rho_dust_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<Tscal> buf_flux_rho_dust_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<Tscal> buf_flux_rho_dust_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<Tscal> buf_flux_rho_dust_face_zm{*oriented_cell_graph.graph_links[izm]};


        NGLink<Tvec> buf_flux_rhov_dust_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<Tvec> buf_flux_rhov_dust_face_xm{*oriented_cell_graph.graph_links[ixm]};
        NGLink<Tvec> buf_flux_rhov_dust_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<Tvec> buf_flux_rhov_dust_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<Tvec> buf_flux_rhov_dust_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<Tvec> buf_flux_rhov_dust_face_zm{*oriented_cell_graph.graph_links[izm]};


        if(solver_config.riemman_config == Rusanov){
            constexpr RiemmanSolverMode mode = Rusanov;
            logger::debug_ln("[AMR Flux]", "compute rusanov xp patch", id);
            compute_fluxes_dir<mode,Tvec, Tscal, Direction::xp>(
                q,
                rho_face_xp.link_count,
                rho_face_xp.link_graph_field,
                rhov_face_xp.link_graph_field,
                rhoe_face_xp.link_graph_field,
                buf_flux_rho_face_xp.link_graph_field,
                buf_flux_rhov_face_xp.link_graph_field,
                buf_flux_rhoe_face_xp.link_graph_field,
                gamma);
        
            logger::debug_ln("[AMR Flux]", "compute rusanov yp patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                q,
                rho_face_yp.link_count,
                rho_face_yp.link_graph_field,
                rhov_face_yp.link_graph_field,
                rhoe_face_yp.link_graph_field,
                buf_flux_rho_face_yp.link_graph_field,
                buf_flux_rhov_face_yp.link_graph_field,
                buf_flux_rhoe_face_yp.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute rusanov zp patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                q,
                rho_face_zp.link_count,
                rho_face_zp.link_graph_field,
                rhov_face_zp.link_graph_field,
                rhoe_face_zp.link_graph_field,
                buf_flux_rho_face_zp.link_graph_field,
                buf_flux_rhov_face_zp.link_graph_field,
                buf_flux_rhoe_face_zp.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute rusanov xm patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                q,
                rho_face_xm.link_count,
                rho_face_xm.link_graph_field,
                rhov_face_xm.link_graph_field,
                rhoe_face_xm.link_graph_field,
                buf_flux_rho_face_xm.link_graph_field,
                buf_flux_rhov_face_xm.link_graph_field,
                buf_flux_rhoe_face_xm.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute rusanov ym patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                q,
                rho_face_ym.link_count,
                rho_face_ym.link_graph_field,
                rhov_face_ym.link_graph_field,
                rhoe_face_ym.link_graph_field,
                buf_flux_rho_face_ym.link_graph_field,
                buf_flux_rhov_face_ym.link_graph_field,
                buf_flux_rhoe_face_ym.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute rusanov zm patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                q,
                rho_face_zm.link_count,
                rho_face_zm.link_graph_field,
                rhov_face_zm.link_graph_field,
                rhoe_face_zm.link_graph_field,
                buf_flux_rho_face_zm.link_graph_field,
                buf_flux_rhov_face_zm.link_graph_field,
                buf_flux_rhoe_face_zm.link_graph_field,
                gamma);
        }else if(solver_config.riemman_config == HLL){
            constexpr RiemmanSolverMode mode = HLL;
            logger::debug_ln("[AMR Flux]", "compute HLL xp patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::xp>(
                q,
                rho_face_xp.link_count,
                rho_face_xp.link_graph_field,
                rhov_face_xp.link_graph_field,
                rhoe_face_xp.link_graph_field,
                buf_flux_rho_face_xp.link_graph_field,
                buf_flux_rhov_face_xp.link_graph_field,
                buf_flux_rhoe_face_xp.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute HLL yp patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                q,
                rho_face_yp.link_count,
                rho_face_yp.link_graph_field,
                rhov_face_yp.link_graph_field,
                rhoe_face_yp.link_graph_field,
                buf_flux_rho_face_yp.link_graph_field,
                buf_flux_rhov_face_yp.link_graph_field,
                buf_flux_rhoe_face_yp.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute HLL zp patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                q,
                rho_face_zp.link_count,
                rho_face_zp.link_graph_field,
                rhov_face_zp.link_graph_field,
                rhoe_face_zp.link_graph_field,
                buf_flux_rho_face_zp.link_graph_field,
                buf_flux_rhov_face_zp.link_graph_field,
                buf_flux_rhoe_face_zp.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute HLL xm patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                q,
                rho_face_xm.link_count,
                rho_face_xm.link_graph_field,
                rhov_face_xm.link_graph_field,
                rhoe_face_xm.link_graph_field,
                buf_flux_rho_face_xm.link_graph_field,
                buf_flux_rhov_face_xm.link_graph_field,
                buf_flux_rhoe_face_xm.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute HLL ym patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                q,
                rho_face_ym.link_count,
                rho_face_ym.link_graph_field,
                rhov_face_ym.link_graph_field,
                rhoe_face_ym.link_graph_field,
                buf_flux_rho_face_ym.link_graph_field,
                buf_flux_rhov_face_ym.link_graph_field,
                buf_flux_rhoe_face_ym.link_graph_field,
                gamma);
            logger::debug_ln("[AMR Flux]", "compute HLL zm patch", id);
            compute_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                q,
                rho_face_zm.link_count,
                rho_face_zm.link_graph_field,
                rhov_face_zm.link_graph_field,
                rhoe_face_zm.link_graph_field,
                buf_flux_rho_face_zm.link_graph_field,
                buf_flux_rhov_face_zm.link_graph_field,
                buf_flux_rhoe_face_zm.link_graph_field,
                gamma);
        }

        if(solver_config.dust_riemann_config == DHLL)
        {
            constexpr DustRiemannSolverMode mode = DHLL;
            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll xp patch", id);
            compute_dust_fluxes_dir<mode,Tvec, Tscal, Direction::xp>(
                q,
                rho_dust_face_xp.link_count,
                rho_dust_face_xp.link_graph_field,
                rhov_dust_face_xp.link_graph_field,
                buf_flux_rho_dust_face_xp.link_graph_field,
                buf_flux_rhov_dust_face_xp.link_graph_field);
        
            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll yp patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                q,
                rho_face_yp.link_count,
                rho_dust_face_yp.link_graph_field,
                rhov_dust_face_yp.link_graph_field,
                buf_flux_rho_dust_face_yp.link_graph_field,
                buf_flux_rhov_dust_face_yp.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll zp patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                q,
                rho_dust_face_zp.link_count,
                rho_dust_face_zp.link_graph_field,
                rhov_dust_face_zp.link_graph_field,
                buf_flux_rho_dust_face_zp.link_graph_field,
                buf_flux_rhov_dust_face_zp.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll xm patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                q,
                rho_dust_face_xm.link_count,
                rho_dust_face_xm.link_graph_field,
                rhov_dust_face_xm.link_graph_field,
                buf_flux_rho_dust_face_xm.link_graph_field,
                buf_flux_rhov_dust_face_xm.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll ym patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                q,
                rho_dust_face_ym.link_count,
                rho_dust_face_ym.link_graph_field,
                rhov_dust_face_ym.link_graph_field,
                buf_flux_rho_dust_face_ym.link_graph_field,
                buf_flux_rhov_dust_face_ym.link_graph_field);
                
            logger::debug_ln("[AMR Flux]", "compute dust rusanov/hll zm patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                q,
                rho_dust_face_zm.link_count,
                rho_dust_face_zm.link_graph_field,
                rhov_dust_face_zm.link_graph_field,
                buf_flux_rho_dust_face_zm.link_graph_field,
                buf_flux_rhov_dust_face_zm.link_graph_field);
        }
        else if(solver_config.dust_riemann_config == HB){
            constexpr DustRiemannSolverMode mode = HB;
            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai xp patch", id);
            compute_dust_fluxes_dir<mode,Tvec, Tscal, Direction::xp>(
                q,
                rho_dust_face_xp.link_count,
                rho_dust_face_xp.link_graph_field,
                rhov_dust_face_xp.link_graph_field,
                buf_flux_rho_dust_face_xp.link_graph_field,
                buf_flux_rhov_dust_face_xp.link_graph_field);
        
            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai yp patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::yp>(
                q,
                rho_face_yp.link_count,
                rho_dust_face_yp.link_graph_field,
                rhov_dust_face_yp.link_graph_field,
                buf_flux_rho_dust_face_yp.link_graph_field,
                buf_flux_rhov_dust_face_yp.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai zp patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::zp>(
                q,
                rho_dust_face_zp.link_count,
                rho_dust_face_zp.link_graph_field,
                rhov_dust_face_zp.link_graph_field,
                buf_flux_rho_dust_face_zp.link_graph_field,
                buf_flux_rhov_dust_face_zp.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai xm patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::xm>(
                q,
                rho_dust_face_xm.link_count,
                rho_dust_face_xm.link_graph_field,
                rhov_dust_face_xm.link_graph_field,
                buf_flux_rho_dust_face_xm.link_graph_field,
                buf_flux_rhov_dust_face_xm.link_graph_field);

            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai ym patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::ym>(
                q,
                rho_dust_face_ym.link_count,
                rho_dust_face_ym.link_graph_field,
                rhov_dust_face_ym.link_graph_field,
                buf_flux_rho_dust_face_ym.link_graph_field,
                buf_flux_rhov_dust_face_ym.link_graph_field);
                
            logger::debug_ln("[AMR Flux]", "compute dust Huang_bai zm patch", id);
            compute_dust_fluxes_dir<mode, Tvec, Tscal, Direction::zm>(
                q,
                rho_dust_face_zm.link_count,
                rho_dust_face_zm.link_graph_field,
                rhov_dust_face_zm.link_graph_field,
                buf_flux_rho_dust_face_zm.link_graph_field,
                buf_flux_rhov_dust_face_zm.link_graph_field);
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
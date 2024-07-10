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

#include "shammath/riemann.hpp"

enum Direction{
    xp = 0,
    xm = 1,
    yp = 2,
    ym = 3,
    zp = 4,
    zm = 5,
};

using RiemmanSolverMode = shammodels::basegodunov::RiemmanSolverMode;

template<class Tvec, RiemmanSolverMode mode, Direction dir>
class FluxCompute{
    public: 

    using Tcons = shammath::ConsState<Tvec>;
    using Tscal =  typename Tcons::Tscal;

    inline static Tcons flux(Tcons cL, Tcons cR, typename Tcons::Tscal gamma){
        if constexpr (mode == RiemmanSolverMode::Rusanov){
            if constexpr (dir == xp){
                return shammath::rusanov_flux_x(cL, cR, gamma);
            }
            if constexpr (dir == yp){
                return shammath::rusanov_flux_y(cL, cR, gamma);
            }
            if constexpr (dir == zp){
                return shammath::rusanov_flux_z(cL, cR, gamma);
            }
            if constexpr (dir == xm){
                return shammath::rusanov_flux_mx(cL, cR, gamma);
            }
            if constexpr (dir == ym){
                return shammath::rusanov_flux_my(cL, cR, gamma);
            }
            if constexpr (dir == zm){
                return shammath::rusanov_flux_mz(cL, cR, gamma);
            }
        }
        if constexpr (mode == RiemmanSolverMode::HLL){
            if constexpr (dir == xp){
                return shammath::hll_flux_x(cL, cR, gamma);
            }
            if constexpr (dir == yp){
                return shammath::hll_flux_y(cL, cR, gamma);
            }
            if constexpr (dir == zp){
                return shammath::hll_flux_z(cL, cR, gamma);
            }
            if constexpr (dir == xm){
                return shammath::hll_flux_mx(cL, cR, gamma);
            }
            if constexpr (dir == ym){
                return shammath::hll_flux_my(cL, cR, gamma);
            }
            if constexpr (dir == zm){
                return shammath::hll_flux_mz(cL, cR, gamma);
            }
        }
    }
};

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_xp(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_xp,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_xp,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_xp,
    sycl::buffer<Tscal> &flux_rho_face_xp,
    sycl::buffer<Tvec> &flux_rhov_face_xp,
    sycl::buffer<Tscal> &flux_rhoe_face_xp,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, xp>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_xp, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_xp, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_xp, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_xp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_xp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_xp, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux xp", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_x = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_x.rho;
            flux_rhov[id_a] = flux_x.rhovel;
            flux_rhoe[id_a] = flux_x.rhoe;
        });
    });
}

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_yp(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_yp,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_yp,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_yp,
    sycl::buffer<Tscal> &flux_rho_face_yp,
    sycl::buffer<Tvec> &flux_rhov_face_yp,
    sycl::buffer<Tscal> &flux_rhoe_face_yp,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, yp>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_yp, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_yp, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_yp, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_yp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_yp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_yp, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux yp", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_y = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_y.rho;
            flux_rhov[id_a] = flux_y.rhovel;
            flux_rhoe[id_a] = flux_y.rhoe;
        });
    });
}

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_zp(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_zp,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_zp,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_zp,
    sycl::buffer<Tscal> &flux_rho_face_zp,
    sycl::buffer<Tvec> &flux_rhov_face_zp,
    sycl::buffer<Tscal> &flux_rhoe_face_zp,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, zp>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_zp, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_zp, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_zp, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_zp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_zp, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_zp, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux zp", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_z = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_z.rho;
            flux_rhov[id_a] = flux_z.rhovel;
            flux_rhoe[id_a] = flux_z.rhoe;
        });
    });
}

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_xm(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_xm,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_xm,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_xm,
    sycl::buffer<Tscal> &flux_rho_face_xm,
    sycl::buffer<Tvec> &flux_rhov_face_xm,
    sycl::buffer<Tscal> &flux_rhoe_face_xm,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, xm>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_xm, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_xm, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_xm, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_xm, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_xm, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_xm, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux xm", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_x = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_x.rho;
            flux_rhov[id_a] = flux_x.rhovel;
            flux_rhoe[id_a] = flux_x.rhoe;
        });
    });
}

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_ym(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_ym,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_ym,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_ym,
    sycl::buffer<Tscal> &flux_rho_face_ym,
    sycl::buffer<Tvec> &flux_rhov_face_ym,
    sycl::buffer<Tscal> &flux_rhoe_face_ym,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, ym>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_ym, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_ym, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_ym, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_ym, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_ym, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_ym, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux ym", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_y = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_y.rho;
            flux_rhov[id_a] = flux_y.rhovel;
            flux_rhoe[id_a] = flux_y.rhoe;
        });
    });
}

template<RiemmanSolverMode mode, class Tvec, class Tscal>
void compute_fluxes_zm(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_face_zm,
    sycl::buffer<std::array<Tvec, 2>> &rhov_face_zm,
    sycl::buffer<std::array<Tscal, 2>> &rhoe_face_zm,
    sycl::buffer<Tscal> &flux_rho_face_zm,
    sycl::buffer<Tvec> &flux_rhov_face_zm,
    sycl::buffer<Tscal> &flux_rhoe_face_zm,
    Tscal gamma) {

    using Flux = FluxCompute<Tvec, mode, zm>;

    q.submit([&, gamma](sycl::handler &cgh) {
        sycl::accessor rho{rho_face_zm, cgh, sycl::read_only};
        sycl::accessor rhov{rhov_face_zm, cgh, sycl::read_only};
        sycl::accessor rhoe{rhoe_face_zm, cgh, sycl::read_only};

        sycl::accessor flux_rho{flux_rho_face_zm, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhov{flux_rhov_face_zm, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor flux_rhoe{flux_rhoe_face_zm, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, link_count, "compute rusanov flux zm", [=](u32 id_a) {
            auto rho_ij  = rho[id_a];
            auto rhov_ij = rhov[id_a];
            auto rhoe_ij = rhoe[id_a];

            using Tconst = shammath::ConsState<Tvec>;

            auto flux_z = Flux::flux(
                Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                gamma);

            flux_rho[id_a]  = flux_z.rho;
            flux_rhov[id_a] = flux_z.rhovel;
            flux_rhoe[id_a] = flux_z.rhoe;
        });
    });
}

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

        /**
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_xp = storage.rho_dust_face_xp.get().get(id);
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_xm = storage.rho_dust_face_xm.get().get(id);
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_yp = storage.rho_dust_face_yp.get().get(id);
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_ym = storage.rho_dust_face_ym.get().get(id);
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_zp = storage.rho_dust_face_zp.get().get(id);
        NGLink<std::vector<std::array<Tscal,2>>> &rho_dust_face_zm = storage.rho_dust_face_zm.get().get(id);

        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_xp =  storage.rhov_dust_face_xp.get().get(id);
        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_xm =  storage.rhov_dust_face_xm.get().get(id);
        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_yp =  storage.rhov_dust_face_yp.get().get(id);
        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_ym =  storage.rhov_dust_face_ym.get().get(id);
        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_zp =  storage.rhov_dust_face_zp.get().get(id);
        NGLink<std::vector<std::array<Tvec,2>>> &rhov_dust_face_zm =  storage.rhov_dust_face_zm.get().get(id);
        **/

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

        /**
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_xp{*oriented_cell_graph.graph_links[ixm]};
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<std::vector<Tscal>> buf_flux_rho_dust_face_zm{*oriented_cell_graph.graph_links[izm]};
        **/

        NGLink<Tvec> buf_flux_rhov_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<Tvec> buf_flux_rhov_face_xm{*oriented_cell_graph.graph_links[ixm]};
        NGLink<Tvec> buf_flux_rhov_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<Tvec> buf_flux_rhov_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<Tvec> buf_flux_rhov_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<Tvec> buf_flux_rhov_face_zm{*oriented_cell_graph.graph_links[izm]};

        /**
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_xp{*oriented_cell_graph.graph_links[ixm]};
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<std::vector<Tvec>> buf_flux_rhov_dust_face_zm{*oriented_cell_graph.graph_links[izm]};
        **/

        NGLink<Tscal> buf_flux_rhoe_face_xp{*oriented_cell_graph.graph_links[ixp]};
        NGLink<Tscal> buf_flux_rhoe_face_xm{*oriented_cell_graph.graph_links[ixm]};
        NGLink<Tscal> buf_flux_rhoe_face_yp{*oriented_cell_graph.graph_links[iyp]};
        NGLink<Tscal> buf_flux_rhoe_face_ym{*oriented_cell_graph.graph_links[iym]};
        NGLink<Tscal> buf_flux_rhoe_face_zp{*oriented_cell_graph.graph_links[izp]};
        NGLink<Tscal> buf_flux_rhoe_face_zm{*oriented_cell_graph.graph_links[izm]};


        if(solver_config.riemman_config == Rusanov){
            constexpr RiemmanSolverMode mode = Rusanov;
            logger::debug_ln("[AMR Flux]", "compute rusanov xp patch", id);
            compute_fluxes_xp<mode>(
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
            compute_fluxes_yp<mode>(
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
            compute_fluxes_zp<mode>(
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
            compute_fluxes_xm<mode>(
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
            compute_fluxes_ym<mode>(
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
            compute_fluxes_zm<mode>(
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
            compute_fluxes_xp<mode>(
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
            compute_fluxes_yp<mode>(
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
            compute_fluxes_zp<mode>(
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
            compute_fluxes_xm<mode>(
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
            compute_fluxes_ym<mode>(
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
            compute_fluxes_zm<mode>(
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

template class shammodels::basegodunov::modules::ComputeFlux<f64_3, i64_3>;
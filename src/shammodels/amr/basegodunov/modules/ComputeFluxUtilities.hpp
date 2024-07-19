// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeDustFluxUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/sycl_utils.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeFlux.hpp"

#include "shammath/riemann.hpp"




using RiemannSolverMode = shammodels::basegodunov::RiemmanSolverMode;
using Direction             = shammodels::basegodunov::modules::Direction;


template<class Tvec, RiemannSolverMode mode, Direction dir>
class FluxCompute{
    public: 

    using Tcons = shammath::ConsState<Tvec>;
    using Tscal =  typename Tcons::Tscal;

    inline static Tcons flux(Tcons cL, Tcons cR, typename Tcons::Tscal gamma){
        if constexpr (mode == RiemannSolverMode::Rusanov){
            if constexpr (dir == Direction::xp){
                return shammath::rusanov_flux_x(cL, cR, gamma);
            }
            if constexpr (dir == Direction::yp){
                return shammath::rusanov_flux_y(cL, cR, gamma);
            }
            if constexpr (dir == Direction::zp){
                return shammath::rusanov_flux_z(cL, cR, gamma);
            }
            if constexpr (dir == Direction::xm){
                return shammath::rusanov_flux_mx(cL, cR, gamma);
            }
            if constexpr (dir == Direction::ym){
                return shammath::rusanov_flux_my(cL, cR, gamma);
            }
            if constexpr (dir == Direction::zm){
                return shammath::rusanov_flux_mz(cL, cR, gamma);
            }
        }
        if constexpr (mode == RiemannSolverMode::HLL){
            if constexpr (dir == Direction::xp){
                return shammath::hll_flux_x(cL, cR, gamma);
            }
            if constexpr (dir == Direction::yp){
                return shammath::hll_flux_y(cL, cR, gamma);
            }
            if constexpr (dir == Direction::zp){
                return shammath::hll_flux_z(cL, cR, gamma);
            }
            if constexpr (dir == Direction::xm){
                return shammath::hll_flux_mx(cL, cR, gamma);
            }
            if constexpr (dir == Direction::ym){
                return shammath::hll_flux_my(cL, cR, gamma);
            }
            if constexpr (dir == Direction::zm){
                return shammath::hll_flux_mz(cL, cR, gamma);
            }
        }
    }
};

 template<RiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
 void compute_fluxes_dir(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal,2>> &rho_face_dir,
    sycl::buffer<std::array<Tvec,2>> &rhov_face_dir,
    sycl::buffer<std::array<Tscal,2>> &rhoe_face_dir,
    sycl::buffer<Tscal> &flux_rho_face_dir,
    sycl::buffer<Tvec> &flux_rhov_face_dir,
    sycl::buffer<Tscal> &flux_rhoe_face_dir, Tscal gamma) {
    using Flux = FluxCompute<Tvec, mode, dir>;
    std::string flux_name = (mode == RiemannSolverMode::HLL) ? "hll" : "Rusanov";
        std::string cur_direction = "";
        if constexpr(dir == Direction::xp)
            cur_direction = "xp";
        if constexpr(dir == Direction::xm)
            cur_direction = "xm";
        if constexpr(dir == Direction::yp)
            cur_direction = "yp";
        if constexpr(dir == Direction::ym)
            cur_direction = "ym";
        if constexpr(dir == Direction::zp)
            cur_direction = "zp";
        if constexpr(dir == Direction::zm)
            cur_direction = "zm";
        
        std::string kernel_name = "compute flux" + flux_name + "flux" + cur_direction;
        const char* _kernel_name = kernel_name.c_str();

        q.submit([&, gamma] (sycl::handler &cgh) {
            sycl::accessor rho{rho_face_dir, cgh, sycl::read_only};
            sycl::accessor rhov{rhov_face_dir, cgh, sycl::read_only};
            sycl::accessor rhoe{rhoe_face_dir, cgh, sycl::read_only};

            sycl::accessor flux_rho{flux_rho_face_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov{flux_rhov_face_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhoe{flux_rhoe_face_dir, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, link_count, _kernel_name, [=](u32 id_a){
                auto rho_ij  = rho[id_a];
                auto rhov_ij = rhov[id_a];
                auto rhoe_ij = rhoe[id_a];

                using Tconst = shammath::ConsState<Tvec>;

                auto flux_dir = Flux::flux(
                    Tconst{rho_ij[0], rhoe_ij[0], rhov_ij[0]},
                    Tconst{rho_ij[1], rhoe_ij[1], rhov_ij[1]},
                    gamma);

                flux_rho[id_a]  = flux_dir.rho;
                flux_rhov[id_a] = flux_dir.rhovel;
                flux_rhoe[id_a] = flux_dir.rhoe; 
            });
        });
    }

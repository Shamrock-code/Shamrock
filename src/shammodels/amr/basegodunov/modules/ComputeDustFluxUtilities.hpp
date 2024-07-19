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
#include "shammath/riemann_dust.hpp"
#include <hipSYCL/sycl/handler.hpp>
#include <hipSYCL/sycl/sycl.hpp>


using DustRiemannSolverMode =shammodels::basegodunov::DustRiemannSolverMode;
using Direction             = shammodels::basegodunov::modules::Direction;


 template<class Tvec, DustRiemannSolverMode mode, Direction dir>
 class DustFluxCompute{
    public:
    using Tcons = shammath::DustConsState<Tvec>;
    using Tscal = typename Tcons::Tscal;

    inline static Tcons flux(Tcons cL, Tcons cR){
        if constexpr (mode == DustRiemannSolverMode::HB){
            if constexpr (dir == Direction::xp){
                    return shammath::huang_bai_flux_x(cL,cR);
            }
            if constexpr (dir == Direction::yp){
                    return shammath::huang_bai_flux_y(cL,cR);
            }
            if constexpr (dir == Direction::zp){
                    return shammath::huang_bai_flux_z(cL,cR);
            }

            if constexpr (dir == Direction::xm){
                    return shammath::huang_bai_flux_mx(cL,cR);
            }
            if constexpr (dir == Direction::ym){
                    return shammath::huang_bai_flux_my(cL,cR);
            }
            if constexpr (dir == Direction::zm){
                    return shammath::huang_bai_flux_mz(cL,cR);
            }
        }
        if constexpr (mode == DustRiemannSolverMode::DHLL){
            if constexpr (dir == Direction::xp){
                    return shammath::d_hll_flux_x(cL,cR);
            }
            if constexpr (dir == Direction::yp){
                    return shammath::d_hll_flux_y(cL,cR);
            }
            if constexpr (dir == Direction::zp){
                    return shammath::d_hll_flux_z(cL,cR);
            }

            if constexpr (dir == Direction::xm){
                    return shammath::d_hll_flux_mx(cL,cR);
            }
            if constexpr (dir == Direction::ym){
                    return shammath::d_hll_flux_my(cL,cR);
            }
            if constexpr (dir == Direction::zm){
                    return shammath::d_hll_flux_mz(cL,cR);
            }
        }
    }
 };


 template<DustRiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
 void compute_dust_fluxes_dir(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal,2>> &rho_dust_face_dir,
    sycl::buffer<std::array<Tvec,2>> &rhov_dust_face_dir,
    sycl::buffer<Tscal> &flux_rho_dust_face_dir,
    sycl::buffer<Tvec> &flux_rhov_dust_face_dir) {

        using Flux = DustFluxCompute<Tvec, mode, dir>;
        std::string flux_name = (mode == DustRiemannSolverMode::DHLL) ? "hll" : "huang_bai";
auto get_dir_name = [&](){
    if constexpr(dir == Direction::xp){
        return "xp";
    }else if constexpr(dir == Direction::xm){
        return "xm";
    }else if constexpr(dir == Direction::yp){
        return "yp";
    }else if constexpr(dir == Direction::ym){
        return "ym";
    }else if constexpr(dir == Direction::zp){
        return "zp";
    }else if constexpr(dir == Direction::zm){
        return "zm";
    }else {
        static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
    }
}
std::string cur_direction = get_dir_name();
        
        std::string kernel_name = "compute flux" + flux_name + "flux" + cur_direction;
        const char* _kernel_name = kernel_name.c_str();
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust{rho_dust_face_dir, cgh, sycl::read_only};
            sycl::accessor rhov_dust{rhov_dust_face_dir,cgh, sycl::read_only};
            
            sycl::accessor flux_rho_dust{flux_rho_dust_face_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_face_dir, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, link_count, _kernel_name, [=](u32 id_a) {
                auto rho_dust_ij  = rho_dust[id_a];
                auto rhov_dust_ij = rhov_dust[id_a];

                using Tcons  = shammath::DustConsState<Tvec>;

                auto flux_dust_dir = Flux::flux(Tcons{rho_dust_ij[0], rhov_dust_ij[0]},
                    Tcons{rho_dust_ij[1]}, rhov_dust_ij[1]);
                
                flux_rhov_dust[id_a] = flux_dust_dir.rho;
                flux_rhov_dust[id_a] = flux_dust_dir.rhovel;
            });
        });

    }

    
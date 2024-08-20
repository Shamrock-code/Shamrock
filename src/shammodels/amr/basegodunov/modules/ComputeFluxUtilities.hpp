// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeFluxUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */


#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeFlux.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shambackends/sycl.hpp"
#include <array>
#include <string>



 using  RiemannSolverMode     = shammodels::basegodunov::RiemmanSolverMode;
 using  DustRiemannSolverMode = shammodels::basegodunov::DustRiemannSolverMode;
 using  Direction             = shammodels::basegodunov::modules::Direction;

 template<class Tvec, RiemannSolverMode mode, Direction dir>
 class FluxCompute{
    public:
        using Tcons = shammath::ConsState<Tvec>;
        using Tprim = shammath::PrimState<Tvec>;
        using Tscal = typename Tcons::Tscal;

        inline static Tcons flux(Tprim pL, Tprim pR, typename Tcons::Tscal gamma)
        {
            Tcons cL = shammath::prim_to_cons(pL, gamma);
            Tcons cR = shammath::prim_to_cons(pR, gamma);

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

 template<class Tvec, DustRiemannSolverMode mode, Direction dir>
 class DustFluxCompute{
    public:
    using Tcons = shammath::DustConsState<Tvec>;
    using Tprim = shammath::DustPrimState<Tvec>;
    using Tscal = typename Tcons::Tscal;

    inline static Tcons dustflux(Tprim pL, Tprim pR){

        Tcons cL = shammath::d_prim_to_cons(pL);
        Tcons cR = shammath::d_prim_to_cons(pR);

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

 template<RiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
 void compute_fluxes_dir(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal,2>> &rho_face_dir,
    sycl::buffer<std::array<Tvec,2>> &vel_face_dir,
    sycl::buffer<std::array<Tscal,2>> &press_face_dir,
    sycl::buffer<Tscal> &flux_rho_face_dir,
    sycl::buffer<Tvec> &flux_rhov_face_dir,
    sycl::buffer<Tscal> &flux_rhoe_face_dir,
    Tscal gamma) {
        
        using Flux = FluxCompute<Tvec,mode,dir>;
        std::string flux_name = (mode == RiemannSolverMode::HLL) ? "hll flux " : "rusanov flux ";
        auto get_dir_name =[&](){
            if constexpr (dir == Direction::xp){return "xp";}
            else if constexpr (dir == Direction::xm){return  "xm";}
            else if constexpr (dir == Direction::yp){return "yp";}
            else if constexpr (dir == Direction::ym){return "ym";}
            else if constexpr (dir == Direction::zp){return "zp";}
            else if constexpr (dir == Direction::zm){return "zm";}
            else {
                static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
            }
            return "";
        };
        std::string cur_direction = get_dir_name();
        std::string kernel_name = (std::string)"compute " + flux_name + cur_direction;
        const char* _kernel_name = kernel_name.c_str();

        q.submit([&, gamma](sycl::handler &cgh) {
            sycl::accessor rho{rho_face_dir, cgh, sycl::read_only};
            sycl::accessor vel{vel_face_dir, cgh, sycl::read_only};
            sycl::accessor press{press_face_dir, cgh, sycl::read_only};

            sycl::accessor flux_rho{flux_rho_face_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov{flux_rhov_face_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhoe{flux_rhoe_face_dir, cgh, sycl::write_only, sycl::no_init};


            shambase::parralel_for(cgh, link_count, _kernel_name, [=](u32 id_a) {
                auto rho_ij = rho[id_a];
                auto vel_ij = vel[id_a];
                auto press_ij =press[id_a];

                using Tprim = shammath::PrimState<Tvec>;
                auto flux_dir = Flux::flux(
                    Tprim{rho_ij[0], press_ij[0], vel_ij[0]},
                    Tprim{rho_ij[1], press_ij[1], vel_ij[1]},
                    gamma);

                    flux_rho[id_a]   = flux_dir.rho;
                    flux_rhov[id_a]  = flux_dir.rhovel;
                    flux_rhoe[id_a]  = flux_dir.rhoe;
            });
        });
    }

template <DustRiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
void dust_compute_fluxes_dir(
    sycl::queue &q,
    u32 link_count,
    sycl::buffer<std::array<Tscal, 2>> &rho_dust_dir,
    sycl::buffer<std::array<Tvec,2>> &vel_dust_dir,
    sycl::buffer<Tscal> &flux_rho_dust_dir,
    sycl::buffer<Tvec> &flux_rhov_dust_dir) {

        using d_Flux = DustFluxCompute<Tvec, mode, dir>;
        std::string flux_name = (mode == DustRiemannSolverMode::DHLL) ? "dust hll flux " : "dust huang-bai flux ";
        auto get_dir_name =[&](){
            if constexpr (dir == Direction::xp){return "xp";}
            else if constexpr (dir == Direction::xm){return  "xm";}
            else if constexpr (dir == Direction::yp){return "yp";}
            else if constexpr (dir == Direction::ym){return "ym";}
            else if constexpr (dir == Direction::zp){return "zp";}
            else if constexpr (dir == Direction::zm){return "zm";}
            else {
                static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
            }
            return "";
        };
        std::string cur_direction = get_dir_name();
        std::string kernel_name = (std::string)"compute " + flux_name + cur_direction;
        const char* _kernel_name = kernel_name.c_str();

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor rho_dust{rho_dust_dir, cgh, sycl::read_only};
            sycl::accessor vel_dust{vel_dust_dir, cgh, sycl::read_only};

            sycl::accessor flux_rho_dust{flux_rho_dust_dir, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor flux_rhov_dust{flux_rhov_dust_dir, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, link_count, _kernel_name, [=](u32 id_a) {
                auto rho_ij = rho_dust[id_a];
                auto vel_ij = vel_dust[id_a];

                using Tprim = shammath::DustPrimState<Tvec>;
                auto flux_dust_dir = d_Flux::dustflux(
                    Tprim{rho_ij[0], vel_ij[0]},
                    Tprim{rho_ij[1], vel_ij[1]}
                );

                flux_rho_dust[id_a]   = flux_dust_dir.rho;
                flux_rhov_dust[id_a]  = flux_dust_dir.rhovel;
            });
        });
    }

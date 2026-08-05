// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file riemann.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamphys/eos.hpp"
#include "shamphys/eos_config.hpp"
#include "shammodels/common/EOSConfig.hpp"
#include <iostream>
namespace shammath {

    template<class Tvec_>
    struct ConsState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, rhoe{};
        Tvec rhovel{};

        const ConsState &operator+=(const ConsState &);
        const ConsState &operator-=(const ConsState &);
        const ConsState &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct PrimState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, press{};
        Tvec vel{};
    };

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator+=(const ConsState<Tvec> &cst) {
        rho += cst.rho;
        rhoe += cst.rhoe;
        rhovel += cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator+(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator-=(const ConsState<Tvec> &cst) {
        rho -= cst.rho;
        rhoe -= cst.rhoe;
        rhovel -= cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator-(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator*=(
        const typename ConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhoe *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const typename ConsState<Tvec>::Tscal factor, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const ConsState<Tvec> &lhs, const typename ConsState<Tvec>::Tscal factor) {
        return ConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec_>
    struct Fluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<ConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec> rhoekin(
        shambase::VecComponent<Tvec> rho, Tvec v) {
        using Tscal    = shambase::VecComponent<Tvec>;
        const Tscal v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        return 0.5 * rho * v2;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec> prim_to_cons(
        const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> cons;

        cons.rho = prim.rho;

        const auto rhoeint = prim.press / (gamma - 1.0);
        cons.rhoe          = rhoeint + rhoekin(prim.rho, prim.vel);

        cons.rhovel[0] = prim.rho * prim.vel[0];
        cons.rhovel[1] = prim.rho * prim.vel[1];
        cons.rhovel[2] = prim.rho * prim.vel[2];

        return cons;
    }

    template<class Tvec, class TgridVec>
    inline constexpr PrimState<Tvec> cons_to_prim(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        PrimState<Tvec> prim;

        prim.rho = cons.rho;

        prim.vel[0] = cons.rhovel[0] / cons.rho;
        prim.vel[1] = cons.rhovel[1] / cons.rho;
        prim.vel[2] = cons.rhovel[2] / cons.rho;

        // const auto rhoeint = cons.rhoe - rhoekin(prim.rho, prim.vel);
        const auto eint = (cons.rhoe - rhoekin(prim.rho, prim.vel))*(1./prim.rho);


        using Tscal              = shambase::VecComponent<Tvec>;
        using SolverConfig =  shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using EOSConfig    =  typename  SolverConfig::EOSConfig;
        using EOS_Isothermal =  typename  EOSConfig::Isothermal;
        using EOS_Adiabatic = typename  EOSConfig::Adiabatic;
        using EOS_Barotropic = typename EOSConfig::Barotropic;

        if ( const EOS_Isothermal *eos_config = std::get_if<EOS_Isothermal>(&actual_eos_config.config)){
            prim.press =  shamphys::EOS_Isothermal<Tscal>::pressure(eos_config->cs, prim.rho);
        }

        else if(const EOS_Adiabatic *eos_config = std::get_if<EOS_Adiabatic>(&actual_eos_config.config)){
            prim.press = shamphys::EOS_Adiabatic<Tscal>::pressure(eos_config->gamma, prim.rho, eint);
        }

        else if ( const EOS_Barotropic * eos_config = std::get_if<EOS_Barotropic>(&actual_eos_config.config)){
            prim.press = shamphys::EOS_Barotropic<Tscal>::pressure(eos_config->rho_critic, prim.rho, eos_config->gamma, eos_config->temp, eos_config->mu );
        }
        return prim;
    }

    template<class Tvec,class TgridVec>
    inline constexpr ConsState<Tvec> hydro_flux_x(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        ConsState<Tvec> flux;

        const PrimState<Tvec> prim = cons_to_prim<Tvec,TgridVec>(cons, gamma, actual_eos_config);

        flux.rho = cons.rhovel[0];

        flux.rhoe = (cons.rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = cons.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = cons.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = cons.rho * prim.vel[0] * prim.vel[2];

        return flux;
    }

    template<class Tvec, class TgridVec>
    inline constexpr shambase::VecComponent<Tvec> sound_speed(
        PrimState<Tvec> prim, shambase::VecComponent<Tvec> gamma,  const  shammodels::EOSConfig<Tvec>& actual_eos_config) {


        using Tscal              = shambase::VecComponent<Tvec>;
        using SolverConfig =  shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using EOSConfig    =  typename  SolverConfig::EOSConfig;
        using EOS_Isothermal =  typename  EOSConfig::Isothermal;
        using EOS_Adiabatic = typename  EOSConfig::Adiabatic;
        using EOS_Barotropic = typename EOSConfig::Barotropic;

        using Tscal =  shambase::VecComponent<Tvec>  ;
        Tscal cs = 0.;

        if ( const EOS_Isothermal *eos_config = std::get_if<EOS_Isothermal>(&actual_eos_config.config)){
           cs = eos_config->cs;
        }
        else if( const EOS_Adiabatic *eos_config = std::get_if<EOS_Adiabatic>(&actual_eos_config.config)){
            cs = shamphys::EOS_Adiabatic<Tscal>::cs_from_p(eos_config->gamma,  prim.rho, prim.press);
        }
       else if ( const EOS_Barotropic * eos_config = std::get_if<EOS_Barotropic>(&actual_eos_config.config)){
            cs =  shamphys::EOS_Barotropic<Tscal>::soundspeed(eos_config->rho_critic, prim.rho, eos_config->gamma, eos_config->temp, eos_config->mu);
       }

       return  cs;
       
    }


    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        Tcons flux;

        const auto primL = cons_to_prim<Tvec,TgridVec>(cL, gamma, actual_eos_config);
        const auto primR = cons_to_prim<Tvec,TgridVec>(cR, gamma, actual_eos_config);

        const auto csL = sound_speed<Tvec,TgridVec>(primL, gamma, actual_eos_config);
        const auto csR = sound_speed<Tvec,TgridVec>(primR, gamma, actual_eos_config);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(primL.vel[0]) + csL), (sham::abs(primR.vel[0]) + csR));

        const auto fL = hydro_flux_x<Tvec,TgridVec>(cL, gamma, actual_eos_config);
        const auto fR = hydro_flux_x<Tvec,TgridVec>(cR, gamma, actual_eos_config);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (cR - cL) * S);
    }

    template<class Tcons>
    inline constexpr Tcons y_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[1];
        cprime.rhovel[1] = -c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_y(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[1];
        cprime.rhovel[1] = c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons z_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_z(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons invert_axis(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[0];
        cprime.rhovel[1] = -c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[2];
        return cprime;
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_y(rusanov_flux_x<Tcons, Tvec, TgridVec>(y_to_x(cL), y_to_x(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_z(rusanov_flux_x<Tcons, Tvec, TgridVec>(z_to_x(cL), z_to_x(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(rusanov_flux_x<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(rusanov_flux_y<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons rusanov_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(rusanov_flux_z<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr auto hll_flux_x(
        const Tcons consL, const Tcons consR, const typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        const auto primL = cons_to_prim<Tvec,TgridVec>(consL, gamma, actual_eos_config);
        const auto primR = cons_to_prim<Tvec,TgridVec>(consR, gamma, actual_eos_config);

        const auto csL = sound_speed<Tvec,TgridVec>(primL, gamma, actual_eos_config);
        const auto csR = sound_speed<Tvec,TgridVec>(primR, gamma, actual_eos_config);

        // Teyssier form
        // const auto S_L = sham::min(primL.vel[0], primR.vel[0]) - sham::max(csL, csR);
        // const auto S_R = sham::max(primL.vel[0], primR.vel[0]) + sham::max(csL, csR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(primL.vel[0] - csL, primR.vel[0] - csR);
        const auto S_R = sham::max(primL.vel[0] + csL, primR.vel[0] + csR);

        const auto fluxL = hydro_flux_x<Tvec,TgridVec>(consL, gamma, actual_eos_config);
        const auto fluxR = hydro_flux_x<Tvec,TgridVec>(consR, gamma, actual_eos_config);

        // Equation (10.26) from Toro 3rd Edition , Springer 2009
        auto hll_flux = [=]() {
            // const auto S_L_upwind = sham::min(S_L, 0.0);
            // const auto S_R_upwind = sham::max(S_R, 0.0);
            // const auto S_norm     = 1.0 / (S_R_upwind - S_L_upwind);
            // return (fluxL * S_R_upwind - fluxR * S_L_upwind
            //         + (consR - consL) * S_R_upwind * S_L_upwind)
            //        * S_norm;

            if (S_L >= 0)
                return fluxL;
            else if (S_R <= 0)
                return fluxR;
            else {
                const auto S_norm = 1.0 / (S_R - S_L);
                return (fluxL * S_R - fluxR * S_L + (consR - consL) * S_R * S_L) * S_norm;
            }
        };

        return hll_flux();
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hll_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_y(hll_flux_x<Tcons, Tvec, TgridVec>(y_to_x(cL), y_to_x(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hll_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_z(hll_flux_x<Tcons, Tvec, TgridVec>(z_to_x(cL), z_to_x(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hll_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hll_flux_x<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hll_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hll_flux_y<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hll_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hll_flux_z<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009.
     *         The wave speeds estimates are based on Bernd Einfeldt (SIAM, 1988), On Godunov-Type
     *          Methods for Gas Dynamics
     * @tparam Tcons
     * @param cL left  conservative state
     * @param cR right conservative state
     * @param gamma adiabatic index
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        Tcons flux;
        using Tscal = typename Tcons::Tscal;

        // const to prim
        const auto primL = cons_to_prim<Tvec,TgridVec>(cL, gamma, actual_eos_config);
        const auto primR = cons_to_prim<Tvec,TgridVec>(cR, gamma, actual_eos_config);

        // sound speeds
        const auto csL = sound_speed<Tvec,TgridVec>(primL, gamma, actual_eos_config);
        const auto csR = sound_speed<Tvec,TgridVec>(primR, gamma, actual_eos_config);

        // Left and right state fluxes
        const auto FL = hydro_flux_x<Tvec,TgridVec>(cL, gamma, actual_eos_config);
        const auto FR = hydro_flux_x<Tvec,TgridVec>(cR, gamma, actual_eos_config);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = primL.vel[0];

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = primR.vel[0];

        /////////////////// Pressure based wave speed estimation //////////////
        // First compute the pressure estimation in the star region using the primitive variable
        // solver
        //
        // Toro from section 9.3 or Equation (10.67).
        //
        // TODO: It will be interresting to implement and test various pressure estimate algorithms
        // such as : / Two-Rarefaction Riemann Solver (TRRS), Two-Shock Riemann Solver (TSRS) and
        // Adaptive / Riemann Solvers(AIRS or ANRS)
        ////////////////////////////////////////////////////////////////////////
        Tscal rho_bar = 0.5 * (rhoL + rhoR);
        Tscal cs_bar  = 0.5 * (csL + csR);
        Tscal p_pvrs  = 0.5 * (pressL + pressR) - 0.5 * (velxR - velxL) * rho_bar * cs_bar;
        // Pressure in the star region estimate
        Tscal press_star = sham::max(0., p_pvrs);

        // Once the pressure in the star region is known, we then estimates the wave speeds
        // following https://ui.adsabs.harvard.edu/abs/1994ShWav...4...25T/abstract or Equations
        // (10.59 - 10.60) from Toro
        Tscal qL = 0, qR = 0;
        if (press_star <= pressL) {
            qL = 1.;
        } else {
            qL = sycl::sqrt(
                1. + (0.5 * (1. + gamma) / (Tscal) gamma) * (press_star / (Tscal) pressL - 1.));
        }

        if (press_star <= pressR) {
            qR = 1.;
        } else {
            qR = sycl::sqrt(
                1. + (0.5 * (1. + gamma) / (Tscal) gamma) * (press_star / (Tscal) pressR - 1.));
        }

        // wave speed Toro from Equation (10.59)
        Tscal SL = velxL - csL * qL;
        Tscal SR = velxR + csR * qR;

        // lagrangian sound speed
        const Tscal var_L = rhoL * (SL - velxL);
        const Tscal var_R = rhoR * (SR - velxR);

        // S* speed estimate
        // Equation (10.37) from Toro 3rd Edition , Springer 2009
        const Tscal S_star
            = (primR.press - primL.press + velxL * var_L - velxR * var_R) / (var_L - var_R);

        // New pressure estimate in the star region as average the pressure estimate at right
        // and left of S_star in the star region
        // Equation (10.42) from Toro 3rd Edition , Springer 2009
        const Tscal press_LR
            = 0.5 * (pressL + pressR + var_L * (S_star - velxL) + var_R * (S_star - velxR));
        Tvec D{1, 0, 0};
        Tcons D_star{0, S_star, D};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        // Tcons cL_star = (SL * cL - FL + press_star * D_star) * (1.0 / (SL - S_star));
        Tcons cL_star = (SL * cL - FL + press_LR * D_star) * (1.0 / (SL - S_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        // Tcons cR_star = (SR * cR - FR + press_star * D_star) * (1.0 / (SR - S_star));
        Tcons cR_star = (SR * cR - FR + press_LR * D_star) * (1.0 / (SR - S_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons FL_star = FL + SL * (cL_star - cL);
        Tcons FR_star = FR + SR * (cR_star - cR);

        // HLLC flux
        auto hllc_flux = [=]() {
            if (SL >= 0) {
                return FL;
            } else if (S_star >= 0) {
                return FL_star;
            } else if (SR >= 0) {
                return FR_star;
            } else
                return FR;
        };

        return hllc_flux();
    }

    /**
     * @brief HLLC flux in the +y direction
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_y(hllc_flux_x<Tcons, Tvec, TgridVec>(y_to_x(cL), y_to_x(cR), gamma, actual_eos_config));
    }
    /**
     * @brief HLLC flux in the +z direction
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return x_to_z(hllc_flux_x<Tcons, Tvec, TgridVec>(z_to_x(cL), z_to_x(cR), gamma, actual_eos_config));
    }

    /**
     * @brief HLLC flux in the -x direction
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hllc_flux_x<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    /**
     * @brief HLLC flux in the -y direction
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hllc_flux_y<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

    /**
     * @brief HLLC flux in the -z direction
     */
    template<class Tcons, class Tvec, class TgridVec>
    inline constexpr Tcons hllc_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma, const shammodels::EOSConfig<Tvec>& actual_eos_config) {
        return invert_axis(hllc_flux_z<Tcons, Tvec, TgridVec>(invert_axis(cL), invert_axis(cR), gamma, actual_eos_config));
    }

} // namespace shammath

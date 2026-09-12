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
 * @file riemann_hllc.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief HLLC Riemann solvers for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009.
     *         The wave speeds estimates are based on Bernd Einfeldt (SIAM, 1988), On Godunov-Type
     *          Methods for Gas Dynamics, using the pressure in the star region estimated through
     *          the primitive variable solver (valid for an adiabatic equation of state).
     *        Computes the flux across a face with unit normal n.
     * @tparam FSpec
     * @param fspec fluid state spec (adiabatic equation of state + flux/wave-speed operations)
     * @param primL left  primitive state
     * @param primR right primitive state
     * @param n face unit normal
     */
    template<FluidStateAdiabaticSpec FSpec>
    inline constexpr typename FSpec::Tcons hllc_adiab_toro_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tvec  = typename FSpec::Tvec;
        using Tcons = typename FSpec::Tcons;

        // fspec.gamma() directly if defined, else gammaL/gammaR each from fspec.gamma(prim)
        const auto [gammaL, gammaR] = get_adiabatic_index_lr(fspec, primL, primR);

        // Conservative form is only needed for the star-state algebra below.
        const Tcons cL = fspec.prim_to_cons(primL);
        const Tcons cR = fspec.prim_to_cons(primR);

        // sound speeds
        const auto csL = fspec.sound_speed(primL);
        const auto csR = fspec.sound_speed(primR);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = fspec.vn(primL, n);

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = fspec.vn(primR, n);

        // Left and right state fluxes
        const auto FL = fspec.flux(primL, n, velxL);
        const auto FR = fspec.flux(primR, n, velxR);

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
                1. + (0.5 * (1. + gammaL) / (Tscal) gammaL) * (press_star / (Tscal) pressL - 1.));
        }

        if (press_star <= pressR) {
            qR = 1.;
        } else {
            qR = sycl::sqrt(
                1. + (0.5 * (1. + gammaR) / (Tscal) gammaR) * (press_star / (Tscal) pressR - 1.));
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
        Tcons D_star{0, S_star, n};

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
        if (SL >= 0) {
            return FL;
        } else if (S_star >= 0) {
            return FL_star;
        } else if (SR >= 0) {
            return FR_star;
        } else
            return FR;
    }

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009, using the
     *        Davis (1988) wave speed estimate instead of the pressure based (p*) estimate, i.e.
     *          SL = min(velxL - csL, velxR - csR)
     *          SR = max(velxL + csL, velxR + csR)
     *        This estimate does not rely on an adiabatic equation of state for the pressure in
     *        the star region and can therefore be used for other equations of state.
     *        Computes the flux across a face with unit normal n.
     * @tparam FSpec
     * @param fspec fluid state spec (equation of state + flux/wave-speed operations)
     * @param primL left  primitive state
     * @param primR right primitive state
     * @param n face unit normal
     */
    template<FluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons hllc_davis_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tvec  = typename FSpec::Tvec;
        using Tcons = typename FSpec::Tcons;

        // Conservative form is only needed for the star-state algebra below.
        const Tcons cL = fspec.prim_to_cons(primL);
        const Tcons cR = fspec.prim_to_cons(primR);

        // sound speeds
        const auto csL = fspec.sound_speed(primL);
        const auto csR = fspec.sound_speed(primR);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = fspec.vn(primL, n);

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = fspec.vn(primR, n);

        // Left and right state fluxes
        const auto FL = fspec.flux(primL, n, velxL);
        const auto FR = fspec.flux(primR, n, velxR);

        // Davis estimate, but we'll see later
        Tscal SL = sham::min(velxL - csL, velxR - csR);
        Tscal SR = sham::max(velxL + csL, velxR + csR);

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
        Tcons D_star{0, S_star, n};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        Tcons cL_star = (SL * cL - FL + press_LR * D_star) * (1.0 / (SL - S_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        Tcons cR_star = (SR * cR - FR + press_LR * D_star) * (1.0 / (SR - S_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons FL_star = FL + SL * (cL_star - cL);
        Tcons FR_star = FR + SR * (cR_star - cR);

        // HLLC flux
        if (SL >= 0) {
            return FL;
        } else if (S_star >= 0) {
            return FL_star;
        } else if (SR >= 0) {
            return FR_star;
        } else
            return FR;
    }

} // namespace shammath

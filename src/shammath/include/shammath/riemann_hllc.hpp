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
     * @param prim_l left  primitive state
     * @param prim_r right primitive state
     * @param n face unit normal
     */
    template<FluidStateAdiabaticSpec FSpec>
    inline constexpr typename FSpec::Tcons hllc_adiab_toro_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tvec  = typename FSpec::Tvec;
        using Tcons = typename FSpec::Tcons;

        // fspec.gamma() directly if defined, else gamma_l/gamma_r each from fspec.gamma(prim)
        const auto [gamma_l, gamma_r] = get_adiabatic_index_lr(fspec, prim_l, prim_r);

        // Conservative form is only needed for the star-state algebra below.
        const Tcons c_l = fspec.prim_to_cons(prim_l);
        const Tcons c_r = fspec.prim_to_cons(prim_r);

        // sound speeds
        const auto cs_l = fspec.sound_speed(prim_l);
        const auto cs_r = fspec.sound_speed(prim_r);

        // Left variables
        const auto rho_l   = prim_l.rho;
        const auto press_l = prim_l.press;
        const auto velx_l  = fspec.vn(prim_l, n);

        // Right variables
        const auto rho_r   = prim_r.rho;
        const auto press_r = prim_r.press;
        const auto velx_r  = fspec.vn(prim_r, n);

        // Left and right state fluxes
        const auto f_l = fspec.flux(prim_l, n, velx_l);
        const auto f_r = fspec.flux(prim_r, n, velx_r);

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
        Tscal rho_bar = 0.5 * (rho_l + rho_r);
        Tscal cs_bar  = 0.5 * (cs_l + cs_r);
        Tscal p_pvrs  = 0.5 * (press_l + press_r) - 0.5 * (velx_r - velx_l) * rho_bar * cs_bar;
        // Pressure in the star region estimate
        Tscal press_star = sham::max(0., p_pvrs);

        // Once the pressure in the star region is known, we then estimates the wave speeds
        // following https://ui.adsabs.harvard.edu/abs/1994ShWav...4...25T/abstract or Equations
        // (10.59 - 10.60) from Toro
        Tscal q_l = 0, q_r = 0;
        if (press_star <= press_l) {
            q_l = 1.;
        } else {
            q_l = sycl::sqrt(
                1.
                + (0.5 * (1. + gamma_l) / (Tscal) gamma_l) * (press_star / (Tscal) press_l - 1.));
        }

        if (press_star <= press_r) {
            q_r = 1.;
        } else {
            q_r = sycl::sqrt(
                1.
                + (0.5 * (1. + gamma_r) / (Tscal) gamma_r) * (press_star / (Tscal) press_r - 1.));
        }

        // wave speed Toro from Equation (10.59)
        Tscal s_l = velx_l - cs_l * q_l;
        Tscal s_r = velx_r + cs_r * q_r;

        // lagrangian sound speed
        const Tscal var_l = rho_l * (s_l - velx_l);
        const Tscal var_r = rho_r * (s_r - velx_r);

        // S* speed estimate
        // Equation (10.37) from Toro 3rd Edition , Springer 2009
        const Tscal s_star
            = (prim_r.press - prim_l.press + velx_l * var_l - velx_r * var_r) / (var_l - var_r);

        // New pressure estimate in the star region as average the pressure estimate at right
        // and left of s_star in the star region
        // Equation (10.42) from Toro 3rd Edition , Springer 2009
        const Tscal press_lr
            = 0.5 * (press_l + press_r + var_l * (s_star - velx_l) + var_r * (s_star - velx_r));
        Tcons d_star{0, s_star, n};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        // Tcons c_l_star = (s_l * c_l - f_l + press_star * d_star) * (1.0 / (s_l - s_star));
        Tcons c_l_star = (s_l * c_l - f_l + press_lr * d_star) * (1.0 / (s_l - s_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        // Tcons c_r_star = (s_r * c_r - f_r + press_star * d_star) * (1.0 / (s_r - s_star));
        Tcons c_r_star = (s_r * c_r - f_r + press_lr * d_star) * (1.0 / (s_r - s_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons f_l_star = f_l + s_l * (c_l_star - c_l);
        Tcons f_r_star = f_r + s_r * (c_r_star - c_r);

        // HLLC flux
        if (s_l >= 0) {
            return f_l;
        } else if (s_star >= 0) {
            return f_l_star;
        } else if (s_r >= 0) {
            return f_r_star;
        } else
            return f_r;
    }

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009, using the
     *        Davis (1988) wave speed estimate instead of the pressure based (p*) estimate, i.e.
     *          s_l = min(velx_l - cs_l, velx_r - cs_r)
     *          s_r = max(velx_l + cs_l, velx_r + cs_r)
     *        This estimate does not rely on an adiabatic equation of state for the pressure in
     *        the star region and can therefore be used for other equations of state.
     *        Computes the flux across a face with unit normal n.
     * @tparam FSpec
     * @param fspec fluid state spec (equation of state + flux/wave-speed operations)
     * @param prim_l left  primitive state
     * @param prim_r right primitive state
     * @param n face unit normal
     */
    template<FluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons hllc_davis_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tvec  = typename FSpec::Tvec;
        using Tcons = typename FSpec::Tcons;

        // Conservative form is only needed for the star-state algebra below.
        const Tcons c_l = fspec.prim_to_cons(prim_l);
        const Tcons c_r = fspec.prim_to_cons(prim_r);

        // sound speeds
        const auto cs_l = fspec.sound_speed(prim_l);
        const auto cs_r = fspec.sound_speed(prim_r);

        // Left variables
        const auto rho_l   = prim_l.rho;
        const auto press_l = prim_l.press;
        const auto velx_l  = fspec.vn(prim_l, n);

        // Right variables
        const auto rho_r   = prim_r.rho;
        const auto press_r = prim_r.press;
        const auto velx_r  = fspec.vn(prim_r, n);

        // Left and right state fluxes
        const auto f_l = fspec.flux(prim_l, n, velx_l);
        const auto f_r = fspec.flux(prim_r, n, velx_r);

        // Davis estimate, but we'll see later
        Tscal s_l = sham::min(velx_l - cs_l, velx_r - cs_r);
        Tscal s_r = sham::max(velx_l + cs_l, velx_r + cs_r);

        // lagrangian sound speed
        const Tscal var_l = rho_l * (s_l - velx_l);
        const Tscal var_r = rho_r * (s_r - velx_r);

        // S* speed estimate
        // Equation (10.37) from Toro 3rd Edition , Springer 2009
        const Tscal s_star
            = (prim_r.press - prim_l.press + velx_l * var_l - velx_r * var_r) / (var_l - var_r);

        // New pressure estimate in the star region as average the pressure estimate at right
        // and left of s_star in the star region
        // Equation (10.42) from Toro 3rd Edition , Springer 2009
        const Tscal press_lr
            = 0.5 * (press_l + press_r + var_l * (s_star - velx_l) + var_r * (s_star - velx_r));
        Tcons d_star{0, s_star, n};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        Tcons c_l_star = (s_l * c_l - f_l + press_lr * d_star) * (1.0 / (s_l - s_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        Tcons c_r_star = (s_r * c_r - f_r + press_lr * d_star) * (1.0 / (s_r - s_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons f_l_star = f_l + s_l * (c_l_star - c_l);
        Tcons f_r_star = f_r + s_r * (c_r_star - c_r);

        // HLLC flux
        if (s_l >= 0) {
            return f_l;
        } else if (s_star >= 0) {
            return f_l_star;
        } else if (s_r >= 0) {
            return f_r_star;
        } else
            return f_r;
    }

} // namespace shammath

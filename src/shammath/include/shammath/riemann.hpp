// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file riemann.hpp
 * @author Leodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 * @brief
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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
    const ConsState<Tvec> &
    ConsState<Tvec>::operator*=(const typename ConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhoe *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec>
    operator*(const typename ConsState<Tvec>::Tscal factor, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec>
    const ConsState<Tvec>
    operator*(const ConsState<Tvec> &lhs, const typename ConsState<Tvec>::Tscal factor) {
        return ConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec_>
    struct Fluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<ConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec>
    rhoekin(shambase::VecComponent<Tvec> rho, Tvec v) {
        using Tscal    = shambase::VecComponent<Tvec>;
        const Tscal v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        return 0.5 * rho * v2;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec>
    prim_to_cons(const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> cons;

        cons.rho = prim.rho;

        const auto rhoeint = prim.press / (gamma - 1.0);
        cons.rhoe          = rhoeint + rhoekin(prim.rho, prim.vel);

        cons.rhovel[0] = prim.rho * prim.vel[0];
        cons.rhovel[1] = prim.rho * prim.vel[1];
        cons.rhovel[2] = prim.rho * prim.vel[2];

        return cons;
    }

    template<class Tvec>
    inline constexpr PrimState<Tvec>
    cons_to_prim(const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        PrimState<Tvec> prim;

        prim.rho = cons.rho;

        prim.vel[0] = cons.rhovel[0] / cons.rho;
        prim.vel[1] = cons.rhovel[1] / cons.rho;
        prim.vel[2] = cons.rhovel[2] / cons.rho;

        const auto rhoeint = cons.rhoe - rhoekin(prim.rho, prim.vel);
        prim.press         = (gamma - 1.0) * rhoeint;

        return prim;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec>
    hydro_flux_x(const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        ConsState<Tvec> flux;

        const PrimState<Tvec> prim = cons_to_prim(cons, gamma);

        flux.rho = cons.rhovel[0];

        flux.rhoe = (cons.rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = cons.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = cons.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = cons.rho * prim.vel[0] * prim.vel[2];

        return flux;
    }

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec>
    sound_speed(PrimState<Tvec> prim, shambase::VecComponent<Tvec> gamma) {
        return sycl::sqrt(gamma * prim.press / prim.rho);
    }

    // template<class Tcons>
    // inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
    //     Tcons flux;

    //     const auto primL = cons_to_prim(cL, gamma);
    //     const auto primR = cons_to_prim(cR, gamma);

    //     const auto csL = sound_speed(primL, gamma);
    //     const auto csR = sound_speed(primR, gamma);

    //     const auto S = sham::max(
    //         sham::max(sham::abs(primL.vel[0] - csL), sham::abs(primR.vel[0] - csR)),
    //         sham::max(sham::abs(primL.vel[0] + csL), sham::abs(primR.vel[0] + csR)));

    //     const auto fL = hydro_flux_x(cL, gamma);
    //     const auto fR = hydro_flux_x(cR, gamma);

    //     return (fL + fR) * 0.5 - (cR - cL) * S;
    // }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        Tcons flux;

        const auto primL = cons_to_prim(cL, gamma);
        const auto primR = cons_to_prim(cR, gamma);

        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(primL.vel[0]) + csL), (sham::abs(primR.vel[0]) + csR));

        const auto fL = hydro_flux_x(cL, gamma);
        const auto fR = hydro_flux_x(cR, gamma);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (cR - cL) * S);
    }

    template<class Tcons>
    inline constexpr Tcons hllc_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        Tcons flux;
        using Tscal = shambase::VecComponent<Tvec>;

        // const to prim
        const auto primL = const_to_prim(cL, gamma);
        const auto primR = const_to_prim(cR, gamma);

        // sound speeds
        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Left and right state fluxes
        const auto FL = hydro_flux_x(cL, gamma);
        const auto FR = hydro_flux_x(cR, gamma);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = primL.vel[0];
        const auto velyL  = primL.vel[1];
        const auto velzL  = primL.vel[2];
        const auto ekinL  = 0.5 * rhoL * (velxL * velxL + velyL * velyL + velzL * velzL);
        const auto etotL  = pressL / (gamma - 1.0) + ekinL;

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = primR.vel[0];
        const auto velyR  = primR.vel[1];
        const auto velzR  = primR.vel[2];
        const auto ekinR  = 0.5 * rhoR * (velxR * velxR + velyR * velyR + velzR * velzR);
        const auto etotR  = pressR / (gamma - 1.0) + ekinR;

        // signal speed
        const Tscal SL = 1;
        const Tscal SR = 1;

        //
        // const Tscal var_L    = rhoL * (SL - velxL);
        // const Tscal var_R    = rhoR * (SR - velxR);
        const Tscal var_L = rhoL * (SL - primL.vel[0]);
        const Tscal var_R = rhoR * (SR - primR.vel[0]);

        // S* speed estimate
        // const Tscal S_star = (pressR - pressL + velxL * var_L - velxR * var_R)/(var_L - var_R);
        const Tscal S_star
            = (primR.press - primL.press + primL.vel[0] * var_L - primR.vel[0] * var_R)
              / (var_L - var_R);
        // P* pression estimate
        // const Tscal press_star = (pressR * var_L - pressL * var_R + (velxL - velxR) * var_L *
        // var_R) / (var_L - var_R);
        const Tscal press_star = (primR.press * var_L - primL.press * var_R
                                  + (primL.vel[0] - primR.vel[0]) * var_L * var_R)
                                 / (var_L - var_R);
        // Left intermediate conservative state in the star region
        //  const Tscal rhoL_star   = var_L / (SL - S_star);
        //  const Tscal velxL_star  = (SL * cL.rhovel[0] - FL.rhovel[0] + press_star) / (SL -
        //  S_star); const Tscal velyL_star  = (SL * cL.rhovel[1] - FL.rhovel[1] ) / (SL - S_star);
        //  const Tscal velzL_star  = (SL * cL.rhovel[2] - FL.rhovel[2] ) / (SL - S_star);
        //  const Tscal etotL_star = (etotL*(SL - velxL) - velxL * pressL + press_star * S_star)/(SL
        //  - S_star);

        Tcons cL_star
            = (SL * cL - FL + press_star * ((Tcons){0, {1, 0, 0}, S_star})) / (SL - S_star);
        // Right intermediate conservative state in the star region
        //  const Tscal rhoR_star  = var_R / (SR - S_star);
        //  const Tscal etotR_star = (etotR*(SR - velxR) - velxR * pressR + press_star * S_star)/(SR
        //  - S_star); const Tscal velxR_star  = (SR * cR.rhovel[0] - FR.rhovel[0] + press_star) /
        //  (SR - S_star); const Tscal velyR_star  = (SR * cR.rhovel[1] - FR.rhovel[1] ) / (SR -
        //  S_star); const Tscal velzR_star  = (SR * cR.rhovel[2] - FR.rhovel[2] ) / (SR - S_star);

        Tcons cR_star
            = (SR * cR - FR + press_star * ((Tcons){0, {1, 0, 0}, S_star})) / (SR - S_star);

        // intemediate Flux in the star region
        Tcons FL_star = FL + SL * (cL_star - cL);
        Tcons FR_star = FR + SR * (cR_star - cR);
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

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(rusanov_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(rusanov_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr auto
    hll_flux_x(const Tcons consL, const Tcons consR, const typename Tcons::Tscal gamma) {
        const auto primL = cons_to_prim(consL, gamma);
        const auto primR = cons_to_prim(consR, gamma);

        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Teyssier form
        // const auto S_L = sham::min(primL.vel[0], primR.vel[0]) - sham::max(csL, csR);
        // const auto S_R = sham::max(primL.vel[0], primR.vel[0]) + sham::max(csL, csR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(primL.vel[0] - csL, primR.vel[0] - csR);
        const auto S_R = sham::max(primL.vel[0] + csL, primR.vel[0] + csR);

        const auto fluxL = hydro_flux_x(consL, gamma);
        const auto fluxR = hydro_flux_x(consR, gamma);

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

    template<class Tcons>
    inline constexpr Tcons hll_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(hll_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(hll_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }
} // namespace shammath

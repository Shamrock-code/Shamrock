// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file forces.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief file containing formulas for sph forces
 */

#include "shambase/numeric_limits.hpp"
#include "shambackends/math.hpp"

namespace shamrock::sph {

    template<class Tscal>
    inline static constexpr Tscal vsig_hydro(
        Tscal abs_v_ab_r_ab, Tscal cs_a, Tscal alpha_av, Tscal beta_av) {
        return alpha_av * cs_a + beta_av * abs_v_ab_r_ab;
        ;
    };

    template<class Tscal>
    inline static constexpr Tscal vsig_u(Tscal P_a, Tscal P_b, Tscal rho_a, Tscal rho_b) {
        Tscal rho_avg = (rho_a + rho_b) * 0.5;
        Tscal abs_dp  = sham::abs(P_a - P_b);
        return sycl::sqrt(abs_dp / rho_avg);
        // Tscal vsig_u = abs_v_ab_r_ab;
    }

    /**
     * @brief \cite Phantom_2018 eq.34, with \f$q^a_{ab} = q^b_{ab} = 0\f$
     *
     * @tparam Tvec
     * @tparam Tscal
     * @param m_b
     * @param rho_a_sq
     * @param rho_b_sq
     * @param P_a
     * @param P_b
     * @param omega_a
     * @param omega_b
     * @param nabla_Wab_ha
     * @param nabla_Wab_hb
     * @return Tvec
     */
    template<class Tvec, class Tscal>
    inline Tvec sph_pressure_symetric(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal P_a,
        Tscal P_b,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb) {

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        // inv_sat(.,eps) mean that if sub_fact_. == 0, we return 0
        Tvec acc_a = ((P_a) *sham::inv_sat_zero(sub_fact_a)) * nabla_Wab_ha;
        Tvec acc_b = ((P_b) *sham::inv_sat_zero(sub_fact_b)) * nabla_Wab_hb;

        return -m_b * (acc_a + acc_b);
    }

    /**
     * @brief \cite Phantom_2018 eq.34
     *
     * @tparam Tvec
     * @tparam Tscal
     * @param m_b
     * @param rho_a_sq
     * @param rho_b_sq
     * @param P_a
     * @param P_b
     * @param omega_a
     * @param omega_b
     * @param qa_ab
     * @param qb_ab
     * @param nabla_Wab_ha
     * @param nabla_Wab_hb
     * @return Tvec
     */
    template<class Tvec, class Tscal>
    inline Tvec sph_pressure_symetric_av(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal P_a,
        Tscal P_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal qa_ab,
        Tscal qb_ab,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb) {
        return sph_pressure_symetric(
            m_b,
            rho_a_sq,
            rho_b_sq,
            P_b + qa_ab,
            P_a + qb_ab,
            omega_a,
            omega_b,
            nabla_Wab_ha,
            nabla_Wab_hb);
    }

    /**
     * @brief \cite Phantom_2018 eq.35
     *
     * @tparam Tvec
     * @tparam Tscal
     * @param P_a
     * @param omega_a_rho_a_inv
     * @param rho_a_inv
     * @param pmass
     * @param v_ab
     * @param grad_W_ab
     * @return Tscal
     */
    template<class Tvec, class Tscal>
    inline Tscal duint_dt_pressure(
        Tscal pmass, Tscal P_a, Tscal inv_omega_a_2_rho_a, Tvec v_ab, Tvec grad_W_ab) {
        return P_a * inv_omega_a_2_rho_a * pmass * sycl::dot(v_ab, grad_W_ab);
    }

    /**
     * @brief \cite Phantom_2018 eq.42
     *
     * @tparam Tvec
     * @tparam Tscal
     * @param pmass
     * @param alpha_u
     * @param vsig_u
     * @param u_ab defined as : \f$u_a - u_b\f$
     * @param Fab_inv_omega_a_rho_a
     * @param Fab_inv_omega_b_rho_b
     * @return Tscal
     */
    template<class Tscal>
    inline Tscal lambda_shock_conductivity(
        Tscal pmass,
        Tscal alpha_u,
        Tscal vsig_u,
        Tscal u_ab,
        Tscal Fab_inv_omega_a_rho_a,
        Tscal Fab_inv_omega_b_rho_b) {
        return pmass * alpha_u * vsig_u * u_ab * Tscal(0.5)
               * (Fab_inv_omega_a_rho_a + Fab_inv_omega_b_rho_b);
    }

    template<class Tvec, class Tscal>
    inline void add_to_derivs_sph_artif_visco_cond(
        Tscal pmass,
        Tscal rho_a_sq,
        Tscal omega_a_rho_a_inv,
        Tscal rho_a_inv,
        Tscal rho_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal Fab_a,
        Tscal Fab_b,
        Tscal u_a,
        Tscal u_b,
        Tscal P_a,
        Tscal P_b,
        Tscal alpha_u,

        Tvec v_ab,
        Tvec r_ab_unit,
        Tscal vsig_u,

        Tscal qa_ab,
        Tscal qb_ab,

        Tvec &dv_dt,
        Tscal &du_dt) {

        Tscal AV_P_a = P_a + qa_ab;
        Tscal AV_P_b = P_b + qb_ab;

        dv_dt += sph_pressure_symetric(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            AV_P_a,
            AV_P_b,
            omega_a,
            omega_b,
            r_ab_unit * Fab_a,
            r_ab_unit * Fab_b);

        // compared to Phantom_2018 eq.35 we move lambda shock artificial viscosity
        // pressure part as just a modified SPH pressure (which is the case already in
        // phantom paper but not written that way)
        du_dt += duint_dt_pressure(
            pmass, AV_P_a, omega_a_rho_a_inv * rho_a_inv, v_ab, r_ab_unit * Fab_a);

        du_dt += lambda_shock_conductivity(
            pmass,
            alpha_u,
            vsig_u,
            u_a - u_b,
            Fab_a * omega_a_rho_a_inv,
            Fab_b / (rho_b * omega_b));
    }

} // namespace shamrock::sph

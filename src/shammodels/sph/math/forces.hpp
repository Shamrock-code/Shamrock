// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file forces.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief file containing formulas for sph forces
 */

#include "shambackends/math.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"

namespace shamrock::sph {

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

        Tvec acc_a = ((P_a) / (sub_fact_a)) * nabla_Wab_ha;
        Tvec acc_b = ((P_b) / (sub_fact_b)) * nabla_Wab_hb;

        if (sub_fact_a == 0)
            acc_a = {0, 0, 0};
        if (sub_fact_b == 0)
            acc_b = {0, 0, 0};

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
     * @brief \cite Phantom_2018 eq.40
     *
     * @tparam Tscal
     * @param rho
     * @param vsig
     * @param v_scal_rhat
     * @return Tscal
     */
    template<class Tscal>
    inline Tscal q_av(Tscal rho, Tscal vsig, Tscal v_scal_rhat) {
        return shambase::sycl_utils::g_sycl_max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
    }
    template<class Tscal>
    inline Tscal q_av_disc(Tscal rho, Tscal h, Tscal rab, Tscal alpha_av, Tscal cs, Tscal vsig, Tscal v_scal_rhat) {
        Tscal q_av_d;
        Tscal rho1 = 1./ rho;
        Tscal rabinv = 1. / (rab);

        if(rab < 1e-9){
            rabinv = 0;
        }

        Tscal prefact = -Tscal(0.5) * rho * sham::abs(rabinv) * h;

        Tscal vsig_disc;
        if (v_scal_rhat < Tscal(0)){
            vsig_disc = vsig;
        } else {
            vsig_disc =  (alpha_av * cs);
        }

        q_av_d =  prefact* vsig_disc * v_scal_rhat;

        return q_av_d;
    }

    enum ViscosityType{
        Standard = 0, Disc = 1
    };


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
        return pmass * alpha_u * vsig_u * u_ab * Tscal(0.5) *
               (Fab_inv_omega_a_rho_a + Fab_inv_omega_b_rho_b);
    }

    template<class Kernel, class Tvec, class Tscal, ViscosityType visco_mode = Standard>
    inline void add_to_derivs_sph_artif_visco_cond(
        Tscal pmass,
        Tvec dr,
        Tscal rab,
        Tscal rho_a,
        Tscal rho_a_sq,
        Tscal omega_a_rho_a_inv,
        Tscal rho_a_inv,
        Tscal rho_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal Fab_a,
        Tscal Fab_b,
        Tvec vxyz_a,
        Tvec vxyz_b,
        Tscal u_a,
        Tscal u_b,
        Tscal P_a,
        Tscal P_b,
        Tscal cs_a,
        Tscal cs_b,
        Tscal alpha_a,
        Tscal alpha_b,
        Tscal h_a,
        Tscal h_b,

        Tscal beta_AV,
        Tscal alpha_u,

        Tvec &dv_dt,
        Tscal &du_dt) {

        Tvec v_ab = vxyz_a - vxyz_b;

        Tvec r_ab_unit = dr / rab;

        if (rab < 1e-9) {
            r_ab_unit = {0, 0, 0};
        }

        // f32 P_b     = cs * cs * rho_b;
        Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
        Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

        /////////////////
        // internal energy update
        //  scalar : f32  | vector : f32_3
        Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
        Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;


        //Tscal vsig_u = abs_v_ab_r_ab;
        Tscal rho_avg = (rho_a + rho_b)*0.5;
        Tscal abs_dp = sham::abs(P_a - P_b);
        Tscal vsig_u = sycl::sqrt(abs_dp/rho_avg);

        Tscal dWab_a = Fab_a;
        Tscal dWab_b = Fab_b;

        Tscal qa_ab;
        Tscal qb_ab;

        if constexpr (visco_mode == Standard){
        qa_ab = q_av(rho_a, vsig_a, v_ab_r_ab);
        qb_ab = q_av(rho_b, vsig_b, v_ab_r_ab);
        }

        if constexpr (visco_mode == Disc){ //from Phantom 2018, eq 120
        qa_ab = q_av_disc(rho_a, h_a, rab, alpha_a, cs_a, vsig_a, v_ab_r_ab);
        qb_ab = q_av_disc(rho_b, h_b, rab, alpha_b, cs_b, vsig_b, v_ab_r_ab);
        } 

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
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b);

        // compared to Phantom_2018 eq.35 we move lambda shock artificial viscosity
        // pressure part as just a modified SPH pressure (which is the case already in
        // phantom paper but not written that way)
        du_dt += duint_dt_pressure(
            pmass, AV_P_a, omega_a_rho_a_inv * rho_a_inv, v_ab, r_ab_unit * dWab_a);

        du_dt += lambda_shock_conductivity(
            pmass,
            alpha_u,
            vsig_u,
            u_a - u_b,
            dWab_a * omega_a_rho_a_inv,
            dWab_b / (rho_b * omega_b));
    }

} // namespace shamrock::sph
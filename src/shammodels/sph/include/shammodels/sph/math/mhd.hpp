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
 * @file mhd.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief file containing formulas for sphmhd forces, evolution of magnetic and divergence cleaning
 * fields.
 */

#include "shambase/numeric_limits.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shamphys/mhd.hpp"
#include "shamunits/Constants.hpp"
#include <tuple>

namespace shamrock::sph::mhd {

    enum MHDType { Ideal = 0, NonIdeal = 1 };

    // mag tension form the Tricco 2023 formula
    template<class Tvec, class Tscal>
    inline Tvec B_dot_grad_W(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tvec B_a,
        Tvec B_b,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb,
        Tscal mu_0) {

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tscal B_dot_grad_W_a = sycl::dot(B_a, nabla_Wab_ha);
        Tscal B_dot_grad_W_b = sycl::dot(B_b, nabla_Wab_hb);

        Tvec acc_a = ((B_dot_grad_W_a) *sham::inv_sat_zero(sub_fact_a)) * B_a / mu_0;
        Tvec acc_b = ((B_dot_grad_W_b) *sham::inv_sat_zero(sub_fact_b)) * B_b / mu_0;

        return -m_b * (acc_a + acc_b);
    }

    // from the Phantom paper formula
    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tvec mag_tension(
        Tscal m_b,
        Tvec B_a,
        Tvec B_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb,
        Tscal sub_fact_a,
        Tscal sub_fact_b,
        Tscal mu_0) {
        Tvec magnetic_tension_term;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {

                Tscal mag_tension_a     = -(1. / mu_0) * B_a[i] * B_a[j];
                Tscal mag_tension_b     = -(1. / mu_0) * B_b[i] * B_b[j];
                Tscal acc_mag_tension_a = mag_tension_a * sham::inv_sat_zero(sub_fact_a);
                Tscal acc_mag_tension_b = mag_tension_b * sham::inv_sat_zero(sub_fact_b);

                magnetic_tension_term[i] += -m_b
                                            * (acc_mag_tension_a * nabla_Wab_ha[j]
                                               + acc_mag_tension_b * nabla_Wab_hb[j]);
            }
        }

        return magnetic_tension_term;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tvec fdivB(
        Tscal m_b,
        Tvec B_a,
        Tvec B_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb,
        Tscal sub_fact_a,
        Tscal sub_fact_b,
        Tscal mu_0) {

        Tscal acc_fdivB_a = sycl::dot(B_a, nabla_Wab_ha) * sham::inv_sat_zero(sub_fact_a);
        Tscal acc_fdivB_b = sycl::dot(B_b, nabla_Wab_hb) * sham::inv_sat_zero(sub_fact_b);
        return -0.5 * B_a * m_b * (acc_fdivB_a + acc_fdivB_b)
               / mu_0; // tested, this is what works best
    }

    template<class Tvec, class Tscal>
    inline Tscal lambda_artes(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal vsigb,
        Tvec B_a,
        Tvec B_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal Fab_a,
        Tscal Fab_b) {

        Tscal B_ab_sq = sycl::dot(B_a - B_b, B_a - B_b);

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tscal acc_a = Fab_a * sham::inv_sat_zero(sub_fact_a);
        Tscal acc_b = Fab_b * sham::inv_sat_zero(sub_fact_b);

        Tscal artres = -0.25 * m_b * vsigb * (acc_a + acc_b) * B_ab_sq;
        return artres;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dB_on_rho_induction_term(
        Tscal m_b, Tscal rho_a_sq, Tvec B_a, Tscal omega_a, Tvec nabla_Wab_ha) {

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal induction_term_no_vab
            = -sham::inv_sat_zero(sub_fact_a) * m_b * sycl::dot(B_a, nabla_Wab_ha);

        return induction_term_no_vab;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tvec dB_on_rho_psi_term(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal psi_a,
        Tscal psi_b,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb) {

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tvec psisubterm_a = ((psi_a) *sham::inv_sat_zero(sub_fact_a)) * nabla_Wab_ha;
        Tvec psisubterm_b = ((psi_b) *sham::inv_sat_zero(sub_fact_b)) * nabla_Wab_hb;

        Tvec psiterm = -m_b * (psisubterm_a + psisubterm_a);

        return psiterm;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dpsi_on_ch_parabolic_propag(
        Tscal m_b, Tscal rho_a, Tvec B_a, Tvec B_b, Tscal omega_a, Tvec nabla_Wab_ha, Tscal ch_a) {

        Tscal sub_fact_a = rho_a * omega_a;
        Tvec B_ab        = (B_a - B_b);

        Tscal divB_a = -(1. * sham::inv_sat_zero(sub_fact_a)) * m_b * sycl::dot(B_ab, nabla_Wab_ha);

        Tscal parabolic_propag = m_b * (ch_a * sham::inv_sat_zero(sub_fact_a))
                                 * sycl::dot(B_ab, nabla_Wab_ha); //-ch_a * divB_a;

        return parabolic_propag;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dpsi_on_ch_parabolic_diff(
        Tscal m_b,
        Tscal rho_a,
        Tvec v_ab,
        Tscal psi_a,
        Tscal omega_a,
        Tvec nabla_Wab_ha,
        Tscal ch_a) {

        Tscal sub_fact_a = 2. * rho_a * omega_a * ch_a;

        Tscal parabolic_diff
            = m_b * (psi_a * sham::inv_sat_zero(sub_fact_a)) * sycl::dot(v_ab, nabla_Wab_ha);

        return parabolic_diff;
    }

    template<class Kernel, class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline void add_to_derivs_spmhd(
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
        Tscal h_a,
        Tscal h_b,

        Tscal alpha_u,

        Tvec B_a,
        Tvec B_b,

        Tscal psi_a,
        Tscal psi_b,

        Tscal mu_0,
        Tscal sigma_mhd,

        Tvec &dv_dt,
        Tscal &du_dt,
        Tvec &dB_on_rho_dt,
        Tscal &dpsi_on_ch_dt,
        Tscal &drho_dt,

        Tvec &mag_pressure,
        Tvec &mag_tension,
        Tvec &gas_pressure,
        Tvec &tensile_corr,

        Tscal &psi_propag,
        Tscal &psi_diff,
        Tscal &psi_cons,

        Tscal &u_pressure_viscous_heating) {

        using namespace shamrock::sph;

        Tvec v_ab      = vxyz_a - vxyz_b;
        Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

        Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
        Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

        Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);
        Tscal vsig_a = shamphys::MHD_physics<Tvec, Tscal>::vsig_MHD(
            v_ab, r_ab_unit, cs_a, B_a, rho_a, mu_0, 1., 1.);
        Tscal vsig_b = shamphys::MHD_physics<Tvec, Tscal>::vsig_MHD(
            v_ab, r_ab_unit, cs_a, B_b, rho_b, mu_0, 1., 1.);

        Tscal dWab_a = Fab_a;
        Tscal dWab_b = Fab_b;

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b * rho_b * omega_b;

        Tscal v_shock_a = shamphys::MHD_physics<Tvec, Tscal>::v_shock(cs_a, B_a, rho_a, mu_0);
        Tscal v_shock_b = shamphys::MHD_physics<Tvec, Tscal>::v_shock(cs_b, B_b, rho_b, mu_0);
        Tscal vsig_B    = shamphys::MHD_physics<Tvec, Tscal>::vsigB(v_ab, r_ab_unit);

        Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
        Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

        Tscal AV_P_a = P_a + qa_ab;
        Tscal AV_P_b = P_b + qb_ab;

        Tvec sum_mag_tension, sum_fdivB    = {0., 0., 0.};
        Tscal sum_psi_propag, sum_psi_diff = 0.;

        // dv/dt = gas_pressure_pishock + magnetic_pressure_term + sum_mag_tension + sum_fdivB;
        // du/dt = pressure_term + viscous_heating + shock_conductivity + artificial_resistivity;
        // d(B/rho)/dt = induction_term + psi_term + dissipation_term
        // d(psi/ch)/dt = psi_propag + psi_diff + psi_cons (not in the SPH sum, hence added in
        // update_derivs)

        // dv/dt terms
        sum_fdivB += fdivB(
            pmass, B_a, B_b, r_ab_unit * dWab_a, r_ab_unit * dWab_b, sub_fact_a, sub_fact_b, mu_0);

        Tvec gas_pressure_pishock = sph::sph_pressure_symetric(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            AV_P_a,
            AV_P_b,
            omega_a,
            omega_b,
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b);

        sum_mag_tension += -B_dot_grad_W(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            B_a,
            B_b,
            omega_a,
            omega_b,
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b,
            mu_0);

        Tvec magnetic_pressure_term = (1. / (2 * mu_0))
                                      * sph::sph_pressure_symetric(
                                          pmass,
                                          rho_a_sq,
                                          rho_b * rho_b,
                                          sycl::dot(B_a, B_a),
                                          sycl::dot(B_b, B_b),
                                          omega_a,
                                          omega_b,
                                          r_ab_unit * dWab_a,
                                          r_ab_unit * dWab_b);

        dv_dt += gas_pressure_pishock + magnetic_pressure_term + sum_mag_tension + sum_fdivB;
        // end dv/dt terms

        // get terms for debugging
        mag_pressure += magnetic_pressure_term;
        gas_pressure += gas_pressure_pishock;
        mag_tension += sum_mag_tension;
        tensile_corr += sum_fdivB;
        // end get terms for debugging

        // du/dt terms
        u_pressure_viscous_heating = sph::duint_dt_pressure(
            pmass, AV_P_a, omega_a_rho_a_inv * rho_a_inv, v_ab, r_ab_unit * dWab_a);

        du_dt += u_pressure_viscous_heating;

        du_dt += sph::lambda_shock_conductivity(
            pmass,
            alpha_u,
            vsig_u,
            u_a - u_b,
            dWab_a * omega_a_rho_a_inv,
            dWab_b / (rho_b * omega_b));

        du_dt += lambda_artes(
            pmass, rho_a_sq, rho_b * rho_b, vsig_B, B_a, B_b, omega_a, omega_b, Fab_a, Fab_b);

        // end du/dt terms

        // d(B/rho)/dt terms
        Tscal rho_diss_term_a = Fab_a * sham::inv_sat_zero(sub_fact_a);
        Tscal rho_diss_term_b = Fab_b * sham::inv_sat_zero(sub_fact_b);

        Tvec dB_on_rho_dissipation_term
            = 0.5 * pmass * (rho_diss_term_a + rho_diss_term_b) * (B_a - B_b) * vsig_B;

        dB_on_rho_dt
            += v_ab * dB_on_rho_induction_term(pmass, rho_a_sq, B_a, omega_a, r_ab_unit * dWab_b);

        dB_on_rho_dt += dB_on_rho_psi_term(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            psi_a,
            psi_b,
            omega_a,
            omega_b,
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b);

        dB_on_rho_dt += dB_on_rho_dissipation_term;

        // end d(B/rho)/dt terms

        // d(psi/ch)/dt terms
        sum_psi_propag = dpsi_on_ch_parabolic_propag(
            pmass, rho_a, B_a, B_b, omega_a, r_ab_unit * dWab_a, v_shock_a);

        sum_psi_diff = dpsi_on_ch_parabolic_diff(
            pmass, rho_a, v_ab, psi_a, omega_a, r_ab_unit * dWab_a, v_shock_a);

        dpsi_on_ch_dt += sum_psi_propag;
        dpsi_on_ch_dt += sum_psi_diff;

        // end d(psi/ch)/dt terms

        // get debugging terms
        psi_propag += sum_psi_propag;
        psi_diff += sum_psi_diff;
        // end get debugging terms

        // for conservative checks
        drho_dt += (1. / omega_a) * pmass * sycl::dot(v_ab, r_ab_unit * dWab_a);
    }

} // namespace shamrock::sph::mhd

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mhd.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief file containing formulas for sphmhd forces, evolution of magnetic and divergence cleaning
 * fields.
 */

#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamunits/Constants.hpp"
#include "shammodels/sph/math/forces.hpp"
#include <tuple>

namespace shamrock::spmhd {

    enum MHDType { Ideal = 0, NonIdeal = 1 };

    template<class Tscal>
    inline Tscal q_av(Tscal rho, Tscal vsig, Tscal v_scal_rhat) {
        return sham::max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
    }

    template<class Tvec, class Tscal>
    inline Tscal duint_dt_pressure_mhd(
        Tscal pmass, Tscal P_a, Tscal inv_omega_a_2_rho_a, Tvec v_ab, Tvec grad_W_ab) {
        Tscal AV_P_a = P_a; //+ qa_ab;
            //Tscal pressure_hydro = sph::duint_dt_pressure(pmass, AV_P_a, inv_omega_a_2_rho_a, v_ab, grad_W_ab);
        Tscal pressure_mhd = P_a * inv_omega_a_2_rho_a * pmass * sycl::dot(v_ab, grad_W_ab);

        return pressure_mhd;//pressure_hydro;
    }

    template<class Tscal>
    inline Tscal lambda_shock_conductivity_no_artres(
        Tscal pmass,
        Tscal alpha_u,
        Tscal vsig_a,
        Tscal vsig_u,
        Tscal u_ab,
        Tscal v_scal_rhat,
        Tscal omega_a_rho_a_inv,
        Tscal Fab_a,
        Tscal Fab_inv_omega_a_rho_a,
        Tscal Fab_inv_omega_b_rho_b) {

        Tscal term1 = -0.5 * pmass * omega_a_rho_a_inv * vsig_a * v_scal_rhat * v_scal_rhat * Fab_a;
        Tscal term2 = pmass * alpha_u * vsig_u * u_ab * Tscal(0.5)
                      * (Fab_inv_omega_a_rho_a + Fab_inv_omega_b_rho_b);

        return term1 + term2;
        //Tscal term2_hydro = sph::lambda_shock_conductivity(pmass, alpha_u, vsig_u, u_ab, Fab_inv_omega_a_rho_a, Fab_inv_omega_b_rho_b);
        //return term2_hydro;
    }

    template<class Tvec, class Tscal>
    inline Tscal vsigB(Tvec v_ab, Tvec r_ab_unit) {
        Tvec v_cross_r = sycl::cross(v_ab, r_ab_unit);
        Tscal vsig_B_a = sycl::sqrt(
            v_cross_r[0] * v_cross_r[0] + v_cross_r[1] * v_cross_r[1]
            + v_cross_r[2] * v_cross_r[2]);
        return vsig_B_a;
    }

    template<class Tvec, class Tscal>
    inline Tscal vsig(
        Tvec v_ab,
        Tvec r_ab_unit,
        Tscal cs_a,
        Tvec B_a,
        Tscal rho_a,
        Tscal mu_0,
        Tscal alpha_av,
        Tscal beta_av) {
        Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
        Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);
        Tscal v_A_a         = sycl::sqrt(sycl::dot(B_a, B_a) / (mu_0 * rho_a));
        Tscal v_a           = sycl::sqrt(cs_a * cs_a + v_A_a * v_A_a);
        Tscal vsig          = alpha_av * v_a + beta_av * abs_v_ab_r_ab;
        ;
        if (vsig < 0) {
            logger::raw_ln("BIG BIG BIG BIG BIG BIG BIG");
            logger::raw_ln("BIG BIG BIG PROBLEM BIG BIG");
            logger::raw_ln("BIG BIG BIG BIG BIG BIG BIG");
        }
        return vsig;
    }

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

        Tvec acc_a = ((B_dot_grad_W_a) / (sub_fact_a)) * B_a / mu_0;
        Tvec acc_b = ((B_dot_grad_W_b) / (sub_fact_b)) * B_b / mu_0;

        if (sub_fact_a == 0)
            acc_a = {0, 0, 0};
        if (sub_fact_b == 0)
            acc_b = {0, 0, 0};

        return -m_b * (acc_a + acc_b);
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal> //, template<class> class SPHKernel
    inline Tvec v_mhd_symetric_tensor_shockterm_fdiv(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tvec v_ab,
        Tvec r_ab_unit,
        Tscal v_scal_rhat,
        Tscal P_a,
        Tscal P_b,
        Tscal cs_a,
        Tscal cs_b,
        Tvec B_a,
        Tvec B_b,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb,
        Tscal mu_0) {

        Tvec vMHD = {0., 0., 0.};

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tscal alpha_av = 1.;
        Tscal beta_av  = 2.;
        Tscal vsig_a   = vsig<Tvec, Tscal>(
            v_ab, r_ab_unit, cs_a, B_a, sycl::sqrt(rho_a_sq), mu_0, alpha_av, beta_av);
        Tscal vsig_b = vsig<Tvec, Tscal>(
            v_ab, r_ab_unit, cs_b, B_b, sycl::sqrt(rho_b_sq), mu_0, alpha_av, beta_av);
        Tscal q_ab_a = q_av(sycl::sqrt(rho_a_sq), vsig_a, v_scal_rhat);
        Tscal q_ab_b = q_av(sycl::sqrt(rho_b_sq), vsig_a, v_scal_rhat);

        Tvec acc_a = (q_ab_a / (sub_fact_a)) * nabla_Wab_ha;
        Tvec acc_b = (q_ab_b / (sub_fact_b)) * nabla_Wab_hb;

        if (sub_fact_a == 0) {
            acc_a = {0, 0, 0};
        }
        if (sub_fact_b == 0) {
            acc_b = {0, 0, 0};
        }

        vMHD += -m_b * (acc_a + acc_b); // shock term

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Tscal Mij_a = -(1. / mu_0) * B_a[i] * B_a[j];
                Tscal Mij_b = -(1. / mu_0) * B_b[i] * B_b[j];
                if (i == j) {
                    Mij_a += P_a
                             + (1. / (2 * mu_0))
                                   * sycl::dot(
                                       B_a, B_a); //(B_a[0]*B_a[0] + B_a[1]*B_a[1] + B_a[2]*B_a[2]);
                    Mij_b += P_b
                             + (1. / (2 * mu_0))
                                   * sycl::dot(B_b, B_b); //(B_b[0]*B_b[0] + B_b[1]*B_b[1] +
                                                          // B_b[2]*B_b[2]); //sycl::pow(B_b, 2)
                }

                Tscal acc_MHD_a = (Mij_a / sub_fact_a) * nabla_Wab_ha[j];
                Tscal acc_MHD_b = (Mij_b / sub_fact_b) * nabla_Wab_hb[j];

                vMHD[i] += -m_b * (acc_MHD_a + acc_MHD_b);
            }
        }
        Tscal acc_fdivB_a = sycl::dot(B_a, nabla_Wab_ha) / sub_fact_a;
        Tscal acc_fdivB_b = sycl::dot(B_b, nabla_Wab_hb) / sub_fact_b;

        Tvec fdivB_a = -B_a * m_b * (acc_fdivB_a + acc_fdivB_b);

        vMHD += fdivB_a;

        return vMHD;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal> //, template<class> class SPHKernel
    inline std::tuple<Tvec, Tvec, Tvec, Tvec> dv_terms(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tvec v_ab,
        Tvec r_ab_unit,
        Tscal v_scal_rhat,
        Tscal P_a,
        Tscal P_b,
        Tscal cs_a,
        Tscal cs_b,
        Tvec B_a,
        Tvec B_b,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_Wab_ha,
        Tvec nabla_Wab_hb,
        Tscal mu_0,
        bool Tricco) {

        Tvec sum_fdivB = {0., 0., 0.};
        Tvec sum_mag_tension = {0., 0., 0.};
        Tvec sum_mag_pressure = {0., 0., 0.};
        Tvec sum_gas_pressure = {0., 0., 0.};

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tvec pressure_term;
        Tvec magnetic_pressure_term;
        Tvec magnetic_tension_term;

        if (Tricco) {
            //pressure_term += - sph::sph_pressure_symetric( m_b,
            //                                            rho_a_sq, rho_b_sq, 
            //                                            P_a, P_b, 
            //                                            omega_a, omega_b, 
            //                                            nabla_Wab_ha, nabla_Wab_hb); // looks ok

            Tscal sub_fact_a = rho_a_sq * omega_a;
            Tscal sub_fact_b = rho_b_sq * omega_b;

            Tvec acc_a = ((P_a) / (sub_fact_a)) * nabla_Wab_ha;
            Tvec acc_b = ((P_b) / (sub_fact_b)) * nabla_Wab_hb;

            if (sub_fact_a == 0)
                acc_a = {0, 0, 0};
            if (sub_fact_b == 0)
                acc_b = {0, 0, 0};

            pressure_term += -m_b * (acc_a + acc_b);

            Tscal B_a_sq = sycl::dot(B_a, B_a);
            Tscal B_b_sq = sycl::dot(B_b, B_b); 

            //magnetic_pressure_term +=  (1. / (2*mu_0)) * sph::sph_pressure_symetric( m_b,
            //                                            rho_a_sq, rho_b_sq, 
            //                                            B_a_sq, B_b_sq, 
            //                                            omega_a, omega_b, 
            //                                            nabla_Wab_ha, nabla_Wab_hb); //looks ok too

            Tvec accB_a = (1. / (2 * mu_0)) * ((B_a_sq) / (sub_fact_a)) * nabla_Wab_ha;
            Tvec accB_b = (1. / (2 * mu_0)) * ((B_b_sq) / (sub_fact_b)) * nabla_Wab_hb;

            if (sub_fact_a == 0)
                accB_a = {0, 0, 0};
            if (sub_fact_b == 0)
                accB_b = {0, 0, 0};

            magnetic_pressure_term += -m_b * (accB_a + accB_b);

            magnetic_tension_term += - B_dot_grad_W(m_b, 
                                                  rho_a_sq, rho_b_sq, 
                                                  B_a, B_b, 
                                                  omega_a, omega_b, 
                                                  nabla_Wab_ha, nabla_Wab_hb, mu_0);

//            for (int i = 0; i < 3; ++i) {
//                for (int j = 0; j < 3; ++j) {
//
//                    Tscal mag_tension_a =   -(1. / mu_0) * B_a[i] * B_a[j];
//                    Tscal mag_tension_b =   -(1. / mu_0) * B_b[i] * B_b[j];
//                    Tscal acc_mag_tension_a = mag_tension_a / sub_fact_a;
//                    Tscal acc_mag_tension_b = mag_tension_b / sub_fact_b;
//                    if (sub_fact_a == 0)
//                        acc_mag_tension_a = 0.;
//                    if (sub_fact_b == 0)
//                        acc_mag_tension_b = 0.;
//
//                    magnetic_tension_term[i]  +=  - m_b * (acc_mag_tension_a * nabla_Wab_ha[j] + acc_mag_tension_b * nabla_Wab_hb[j]);
//                
//                }
//            }
        }
        else { // Price
            for (int i = 0; i < 3; ++i) {
                Tscal mag_pressure_a = (1. / (2 * mu_0)) * sycl::dot(B_a, B_a); 
                Tscal mag_pressure_b = (1. / (2 * mu_0)) * sycl::dot(B_b, B_b); 
                Tscal acc_mag_pressure_a = mag_pressure_a / sub_fact_a;
                Tscal acc_mag_pressure_b = mag_pressure_b / sub_fact_b;
                if (sub_fact_a == 0)
                    acc_mag_pressure_a = 0.;
                if (sub_fact_b == 0)
                    acc_mag_pressure_b = 0.;
                magnetic_pressure_term[i] += - m_b * (acc_mag_pressure_a * nabla_Wab_ha[i] + acc_mag_pressure_b * nabla_Wab_hb[i]);

                for (int j = 0; j < 3; ++j) {

                    Tscal mag_tension_a =   -(1. / mu_0) * B_a[i] * B_a[j];
                    Tscal mag_tension_b =   -(1. / mu_0) * B_b[i] * B_b[j];
                    Tscal acc_mag_tension_a = mag_tension_a / sub_fact_a;
                    Tscal acc_mag_tension_b = mag_tension_b / sub_fact_b;
                    if (sub_fact_a == 0)
                        acc_mag_tension_a = 0.;
                    if (sub_fact_b == 0)
                        acc_mag_tension_b = 0.;

                    magnetic_tension_term[i]  +=  - m_b * (acc_mag_tension_a * nabla_Wab_ha[j] + acc_mag_tension_b * nabla_Wab_hb[j]);

                }

                Tscal acc_gas_pressure_a = P_a / sub_fact_a;
                Tscal acc_gas_pressure_b = P_b / sub_fact_b;
                if (sub_fact_a == 0)
                    acc_gas_pressure_a = 0.;
                if (sub_fact_b == 0)
                    acc_gas_pressure_b = 0.;
                pressure_term[i] += - m_b * (acc_gas_pressure_a * nabla_Wab_ha[i] + acc_gas_pressure_b * nabla_Wab_hb[i]);
        }
        }


        Tscal acc_fdivB_a = sycl::dot(B_a, nabla_Wab_ha) / sub_fact_a;
        Tscal acc_fdivB_b = sycl::dot(B_b, nabla_Wab_hb) / sub_fact_b;

        if (sub_fact_a == 0)
            acc_fdivB_a = 0.;
        if (sub_fact_b == 0)
            acc_fdivB_b = 0.;

        Tvec fdivB_a = - 0.5 * B_a * m_b * (acc_fdivB_a + acc_fdivB_b) / mu_0; // tested, this is what works best

        sum_gas_pressure += pressure_term;
        sum_mag_pressure += magnetic_pressure_term;
        sum_mag_tension  += magnetic_tension_term;
        sum_fdivB        += fdivB_a;

        return std::make_tuple(sum_gas_pressure, sum_mag_pressure, sum_mag_tension, sum_fdivB);
    }
    template<class Tvec, class Tscal>
    inline Tscal lambda_artes(
        Tscal m_b,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tvec v_ab,
        Tvec r_ab_unit,
        Tvec B_a,
        Tvec B_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal Fab_a,
        Tscal Fab_b) {

        Tscal vsigb   = vsigB<Tvec, Tscal>(v_ab, r_ab_unit); // same for a and b
        Tscal B_ab_sq = sycl::dot(B_a - B_b, B_a - B_b);

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;

        Tscal acc_a = Fab_a / sub_fact_a;
        Tscal acc_b = Fab_b / sub_fact_b;

        if (sub_fact_a == 0) {
            acc_a = 0;
        }
        if (sub_fact_b == 0) {
            acc_b = 0;
        }

        Tscal artres = -0.25 * m_b * vsigb * (acc_a + acc_b) * B_ab_sq;
        return artres;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dB_on_rho_induction_term(
        Tscal m_b, Tscal rho_a_sq, Tvec B_a, Tscal omega_a, Tvec nabla_Wab_ha) {

        Tscal sub_fact_a            = rho_a_sq * omega_a;
        Tscal induction_term_no_vab = -(1. / sub_fact_a) * m_b * sycl::dot(B_a, nabla_Wab_ha);

        if (sub_fact_a == 0) {
            induction_term_no_vab = 0.;
        }

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

        Tvec psisubterm_a = ((psi_a) / (sub_fact_a)) * nabla_Wab_ha;
        Tvec psisubterm_b = ((psi_b) / (sub_fact_b)) * nabla_Wab_hb;

        if (sub_fact_a == 0) {
            psisubterm_a = {0, 0, 0};
        }
        if (sub_fact_b == 0) {
            psisubterm_b = {0, 0, 0};
        }

        Tvec psiterm = -m_b * (psisubterm_a + psisubterm_a);

        return psiterm;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dpsi_on_ch_parabolic_propag(
        Tscal m_b, Tscal rho_a, Tvec B_a, Tvec B_b, Tscal omega_a, Tvec nabla_Wab_ha, Tscal ch_a) {

        Tscal sub_fact_a = rho_a * omega_a;
        Tvec B_ab        = (B_a - B_b);

        Tscal divB_a = -(1. / sub_fact_a) * m_b * sycl::dot(B_ab, nabla_Wab_ha);

        Tscal parabolic_propag
            =  m_b * (ch_a / sub_fact_a) *  sycl::dot(B_ab, nabla_Wab_ha); //-ch_a * divB_a;

        if (sub_fact_a == 0) {
            parabolic_propag = 0;
        }

        return parabolic_propag;
    }

    template<class Tvec, class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dpsi_on_ch_parabolic_diff(
        Tscal m_b,
        Tscal rho_a,
        Tvec v_a,
        Tvec v_b,
        Tscal psi_a,
        Tscal omega_a,
        Tvec nabla_Wab_ha,
        Tscal ch_a) {

        Tscal sub_fact_a = 2. * rho_a * omega_a * ch_a;
        Tvec v_ab        = v_a - v_b;

        Tscal parabolic_diff = m_b * (psi_a / sub_fact_a) * sycl::dot(v_ab, nabla_Wab_ha);

        if (sub_fact_a == 0) {
            parabolic_diff = 0;
        }

        return parabolic_diff;
    }

    template<class Tscal, MHDType MHD_mode = Ideal>
    inline Tscal dpsi_on_ch_conservation(Tscal h_a, Tscal psi_a, Tscal ch_a, Tscal sigma_mhd) {
        return psi_a * 1.0 / h_a;
    }

    template<class Tvec, class Tscal, template<class> class SPHKernel, MHDType MHD_mode = Ideal>
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
        
        Tvec &mag_pressure,
        Tvec &mag_tension,
        Tvec &gas_pressure,
        Tvec &tensile_corr,

        Tscal &psi_propag,
        Tscal &psi_diff,
        Tscal &psi_cons,

        Tscal &u_mhd) {

        using namespace shamrock::sph;
        Tvec v_ab      = vxyz_a - vxyz_b;
        Tvec r_ab_unit = dr / rab;

        if (rab < 1e-9) {
            r_ab_unit = {0, 0, 0};
        }

        Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
        Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

        // Tscal vsig_u = abs_v_ab_r_ab;
        Tscal rho_avg = (rho_a + rho_b) * 0.5;
        Tscal abs_dp  = sham::abs(P_a - P_b);
        Tscal vsig_u  = sycl::sqrt(abs_dp / rho_avg);
        Tscal vsig_a  = vsig(v_ab, r_ab_unit, cs_a, B_a, rho_a, mu_0, 1., 1.);

        Tscal dWab_a = Fab_a;
        Tscal dWab_b = Fab_b;

        // from Phantom 2018, eq 120

        // using Kernel             = SPHKernel<Tscal>;
        // using Config  = shammodels::sph::SolverConfig<Tvec, SPHKernel>;
        // Config config;

        // Tscal mu_0 = config.get_constant_mu_0();
        Tscal v_alfven_a = sycl::sqrt(sycl::dot(B_a, B_a) / (mu_0 * rho_a));
        Tscal v_alfven_b = sycl::sqrt(sycl::dot(B_b, B_b) / (mu_0 * rho_b));
        Tscal v_shock_a  = sycl::sqrt(sycl::pow(cs_a, 2) + sycl::pow(v_alfven_a, 2));
        Tscal v_shock_b  = sycl::sqrt(sycl::pow(cs_b, 2) + sycl::pow(v_alfven_b, 2));

        Tvec v_cross_r = sycl::cross(v_ab, r_ab_unit);
        Tscal vsig_B   = sycl::sqrt(
            v_cross_r[0] * v_cross_r[0] + v_cross_r[1] * v_cross_r[1]
            + v_cross_r[2] * v_cross_r[2]);

        constexpr bool debug_eq_dump = true;
        if (debug_eq_dump) {
            bool Tricco = true; 
            Tvec sum_gas_pressure, sum_mag_pressure, sum_mag_tension, sum_fdivB = {0., 0., 0.};
            std::tie(sum_gas_pressure, sum_mag_pressure, sum_mag_tension, sum_fdivB) = dv_terms(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            v_ab,
            r_ab_unit,
            v_ab_r_ab,
            P_a,
            P_b,
            cs_a,
            cs_b,
            B_a,
            B_b,
            omega_a,
            omega_b,
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b,
            mu_0);

            dv_dt += sum_gas_pressure + sum_mag_pressure + sum_mag_tension + sum_fdivB;
            mag_pressure += sum_mag_pressure;
            gas_pressure += sum_gas_pressure;
            mag_tension += sum_mag_tension;
            tensile_corr += sum_fdivB;

        }
        else {
            dv_dt += v_mhd_symetric_tensor_shockterm_fdiv(
            pmass,
            rho_a_sq,
            rho_b * rho_b,
            v_ab,
            r_ab_unit,
            v_ab_r_ab,
            P_a,
            P_b,
            cs_a,
            cs_b,
            B_a,
            B_b,
            omega_a,
            omega_b,
            r_ab_unit * dWab_a,
            r_ab_unit * dWab_b,
            mu_0);
        }

        // compared to Phantom_2018 eq.35 we move lambda shock artificial viscosity
        // pressure part as just a modified SPH pressure (which is the case already in
        // phantom paper but not written that way)
        //Tscal qa_ab = q_av(sycl::sqrt(rho_a_sq), vsig_a, v_ab_r_ab);

        u_mhd = duint_dt_pressure_mhd(
            pmass, P_a, omega_a_rho_a_inv * rho_a_inv, v_ab, r_ab_unit * dWab_a);

        du_dt += u_mhd;

        //du_dt += lambda_shock_conductivity_no_artres(
        //    pmass,
        //    alpha_u,
        //    vsig_a,
        //    vsig_u,
        //    u_a - u_b,
        //    abs_v_ab_r_ab,
        //    omega_a_rho_a_inv,
        //    Fab_a,
        //    dWab_b / (rho_a * omega_a),
        //    dWab_b / (rho_b * omega_b));

        //du_dt += lambda_artes(
        //    pmass,
        //    rho_a_sq,
        //    rho_b * rho_b,
        //    v_ab,
        //    r_ab_unit,
        //    B_a,
        //    B_b,
        //    omega_a,
        //    omega_b,
        //    Fab_a,
        //    Fab_b);
 

        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b * rho_b * omega_b;

        Tscal rho_diss_term_a = Fab_a / sub_fact_a;
        Tscal rho_diss_term_b = Fab_b / sub_fact_b;

        if (sub_fact_a == 0) {
            rho_diss_term_a = 0;
        }
        if (sub_fact_b == 0) {
            rho_diss_term_b = 0;
        }

        //Tvec dB_on_rho_dissipation_term
        //    = -0.5 * pmass * (rho_diss_term_a + rho_diss_term_b) * (B_a - B_b) * vsig_B;

        dB_on_rho_dt
            += v_ab * dB_on_rho_induction_term(pmass, rho_a_sq, B_a, omega_a, r_ab_unit * dWab_b);

        //dB_on_rho_dt += dB_on_rho_psi_term(
        //    pmass,
        //    rho_a_sq,
        //    rho_b * rho_b,
        //    psi_a,
        //    psi_b,
        //    omega_a,
        //    omega_b,
        //    r_ab_unit * dWab_a,
        //    r_ab_unit * dWab_b);

        //dB_on_rho_dt += dB_on_rho_dissipation_term;

        psi_propag += dpsi_on_ch_parabolic_propag(
            pmass, rho_a, B_a, B_b, omega_a, r_ab_unit * dWab_a, v_shock_a);

        psi_diff += dpsi_on_ch_parabolic_diff(
            pmass, rho_a, vxyz_a, vxyz_b, psi_a, omega_a, r_ab_unit * dWab_a, v_shock_a);

        psi_cons += dpsi_on_ch_conservation(h_a, psi_a, v_shock_a, sigma_mhd);

        dpsi_on_ch_dt += psi_propag;
        dpsi_on_ch_dt += psi_diff;
        dpsi_on_ch_dt += - psi_cons;
    }

} // namespace shamrock::spmhd

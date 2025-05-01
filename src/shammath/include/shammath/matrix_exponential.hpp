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
 * @file matrix_exponential.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "matrix_op.hpp"
#include "shambackends/sycl.hpp"

namespace shammath {

    /**
     *
     *  %%% ALGORITHM GSQT (GENERAL SCALING AND SQUARING TAYLOR ALGORITHM) %%%
     *     %INPUT:  An n x n matrix A, preprocessed if appropriate
     *     %        K, the maximum allowed number of matrix products;
     *     %        {m_k}, k = 1 : K, the orders of the associated polynomials
     *                         Step 1:
     * Execute Algorithm Order-scale, which selects the order and scaling parameters of Taylor
     * polynomial Step 2: Execute Algorithm Taylor-eval to evaluate the Taylor polynomial in the
     * scaled matrix Step 3: Execute the appropriate number of squaring steps of Taylor polynomial.
     */

    /**
     * @brief precomputed optimal Taylor's polynomial orders
     */
    inline constexpr auto sequence_mk() {
        std::array<i32, 9> seq = {0};
        seq[0]                 = 2;
        seq[1]                 = 4;
        seq[2]                 = 6;
        seq[3]                 = 9;
        seq[4]                 = 12;
        seq[5]                 = 16;
        seq[6]                 = 20;
        seq[7]                 = 25;
        seq[8]                 = 30;

        return seq;
    }

    /**
     * @brief precomputed optimal Paterson-Stockmeyer intergers (it's used to compute the matrix
     * power)
     */
    inline constexpr auto sequence_qk() {
        std::array<i32, 9> seq = {0};
        seq[0]                 = 1;
        seq[1]                 = 2;
        seq[2]                 = 2;
        seq[3]                 = 3;
        seq[4]                 = 3;
        seq[5]                 = 4;
        seq[6]                 = 4;
        seq[7]                 = 5;
        seq[8]                 = 5;
        return seq;
    }

    /**
     * @brief precomputed optimal Paterson-Stockmeyer polynomial degrees
     */
    inline constexpr auto sequence_rk() {
        std::array<i32, 9> seq = {0};
        seq[0]                 = 2;
        seq[1]                 = 2;
        seq[2]                 = 3;
        seq[3]                 = 3;
        seq[4]                 = 4;
        seq[5]                 = 4;
        seq[6]                 = 5;
        seq[7]                 = 5;
        seq[8]                 = 6;
        return seq;
    }

    /**
     * @brief precomputed optimal sequence based on backward error analysis
     */
    inline constexpr auto sequence_theta_mk() {
        std::array<f64, 9> seq = {0};
        seq[0]                 = 2.5810e-8;
        seq[1]                 = 3.3972e-4;
        seq[2]                 = 9.0657e-3;
        seq[3]                 = 8.9578e-2;
        seq[4]                 = 2.9962e-1;
        seq[5]                 = 7.80e-1;
        seq[6]                 = 1.4383;
        seq[7]                 = 2.4286;
        seq[8]                 = 3.5397;
        return seq;
    }

    /**
     * @brief precomputed optimal sequence based on backward error analysis
     */
    inline constexpr auto sequence_nheta_mk() {
        std::array<f64, 9> seq = {0};
        seq[0]                 = 8.7334e-6;
        seq[1]                 = 1.6778e-3;
        seq[2]                 = 1.7720e-3;
        seq[3]                 = 1.1354e-1;
        seq[4]                 = 3.2690e-1;
        seq[5]                 = 7.8738e-1;
        seq[6]                 = 1.4383;
        seq[7]                 = 2.42860;
        seq[8]                 = 3.5397;
        return seq;
    }

    /**
     * @brief 1/(i!)
     */
    inline constexpr auto define_bexp_coef() {
        std::array<f64, 30> coefs = {0};
        coefs[0]                  = 1.0;
        coefs[1]                  = 1.0;
        coefs[2]                  = 0.5;
        coefs[3]                  = 0.16666666666666666;
        coefs[4]                  = 0.041666666666666664;
        coefs[5]                  = 0.008333333333333333;
        coefs[6]                  = 0.001388888888888889;
        coefs[7]                  = 0.0001984126984126984;
        coefs[8]                  = 2.48015873015873e-05;
        coefs[9]                  = 2.7557319223985893e-06;
        coefs[10]                 = 2.755731922398589e-07;
        coefs[11]                 = 2.505210838544172e-08;
        coefs[12]                 = 2.08767569878681e-09;
        coefs[13]                 = 1.6059043836821613e-10;
        coefs[14]                 = 1.1470745597729725e-11;
        coefs[15]                 = 7.647163731819816e-13;
        coefs[16]                 = 4.779477332387385e-14;
        coefs[17]                 = 2.8114572543455206e-15;
        coefs[18]                 = 1.5619206968586225e-16;
        coefs[19]                 = 8.22063524662433e-18;
        coefs[20]                 = 4.110317623312165e-19;
        coefs[21]                 = 1.9572941063391263e-20;
        coefs[22]                 = 8.896791392450574e-22;
        coefs[23]                 = 3.868170170630684e-23;
        coefs[24]                 = 1.6117375710961184e-24;
        coefs[25]                 = 6.446950284384474e-26;
        coefs[26]                 = 2.4795962632247976e-27;
        coefs[27]                 = 9.183689863795546e-29;
        coefs[28]                 = 3.279889237069838e-30;
        coefs[29]                 = 1.1309962886447716e-31;

        return coefs;
    }

    /**
     * @brief this function compute the Taylor's polynomial order (m_star)
     * the optimal number of matrix product during the taylor evaluation step(k_star)
     * and the optimal scaling factor (s_star)
     * @param K maximum number of matrix product allow
     * @param seq_mk precomputed set of Polynomial order
     * @param seq_theta_mk precomputed set of parameters
     * @param A the matrix
     * @param size_A the matrix A size
     * @param k_star the optimal number of matrix product during the taylor evaluation step
     * @param m_star the Taylor's polynomial order
     * @param s_star the optimal scaling factor
     */
    template<class T, class Extents1, class Layout1, class Accessor1>
    inline void order_scale(
        const i32 K,
        std::array<i32, 9> &seq_mk,
        std::array<f64, 9> &seq_theta_mk,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &A,
        const size_t size_A,
        i32 &k_star,
        i32 &m_star,
        i32 &s_star) {
        m_star = seq_mk[K - 1];
        k_star = K;
        s_star = 0;

        T norm_A = 0;
        mat_L1_norm<T>(A, norm_A);

        i32 s_tilde = static_cast<i32>(sycl::ceil(sham::max(
            static_cast<f64>(0.0), static_cast<f64>(sycl::log2(norm_A / seq_theta_mk[K - 1])))));
        s_star      = s_tilde;
        i32 k       = 2;
        bool cond   = false;
        for (; k < (K + 1) && !cond; k++) {
            cond   = (norm_A <= seq_theta_mk[k - 1]) && (norm_A <= seq_theta_mk[K - 1]);
            m_star = cond * seq_mk[k - 1] + !cond * m_star;
            k_star = cond * k + !cond * k_star;
        }
        i32 k_choice = sham::min(K, k); // if we break the preceding loop then use k else K
        auto ld_7    = [&](i32 s_val) {
            k_star = k_choice - 1;
            s_star = sham::max(static_cast<i32>(0), s_val);
            m_star = seq_mk[k_star - 1];
        };

        auto ld_8 = [&](i32 s_val) {
            k_star = k_choice - 2;
            s_star = sham::max(static_cast<i32>(1), s_val + 1);
            m_star = seq_mk[k_star - 1];
        };

        auto ld_9 = [&](i32 s_val) {
            k_star = k_choice - 3;
            s_star = sham::max(static_cast<i32>(2), s_val + 2);
            m_star = seq_mk[k_star - 1];
        };

        f64 cmp_1 = norm_A / (1 << s_tilde);

        i32 val_2 = ((k_choice >= 8) && (cmp_1 <= 2 * seq_theta_mk[k_choice - 3])) * 2;
        i32 val_3 = ((k_choice >= 9) && (cmp_1 <= 4 * seq_theta_mk[k_choice - 4])) * 3;
        i32 val_1 = ((k_choice >= 7) && (cmp_1 <= seq_theta_mk[k_choice - 2]));

        i32 val = sham::max(val_1, sham::max(val_2, val_3));

        auto process_val = [&](int val) {
            if (val == 1) {
                ld_7(s_tilde);
            } else if (val == 2) {
                ld_8(s_tilde);
            } else if (val == 3) {
                ld_9(s_tilde);
            }
        };

        process_val(val);
    }

    /**
     * @brief This function compute the Taylor polynomial up to order m_star
     * @param q Paterson-Stockmeyer interger (it's used to compute the matrix power)
     * @param r Paterson-Stockmeyer polynomial degree
     * @param bi_seq sequence of coef needed for Paterson-Stockmeyer coefficient B_k
     * @param size size of matrices
     * @param A input matrix
     * @param F output matrix
     * @param B,I,Id matrices for intermediate computations
     */
    template<
        typename T,
        class U,
        class Extents1,
        class Extents2,
        class Extents3,
        class Extents4,
        class Extents5,
        class Layout1,
        class Layout2,
        class Layout3,
        class Layout4,
        class Layout5,
        class Accessor1,
        class Accessor2,
        class Accessor3,
        class Accessor4,
        class Accessor5>
    inline void taylor_eval(
        const i32 r,
        const i32 q,
        std::array<f64, 30> &bi_seq,
        const size_t size,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &A,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &F,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &B,
        const std::mdspan<T, Extents4, Layout4, Accessor4> &I,
        const std::mdspan<T, Extents5, Layout5, Accessor5> &Id) {
        mat_set_nul<T>(F);

        for (auto k = r - 1; k >= 0; k--) {
            mat_set_identity<T>(I);
            mat_set_identity<T>(Id);
            mat_set_nul<T>(B);
            i32 cc = 0;

            for (auto j = 1; j <= q; j++) {
                mat_copy<T>(I, Id);
                mat_gemm<T, U>(1, A, Id, 0, I);
                cc = q * k + j;
                mat_axpy_beta<T, U>(bi_seq[cc], I, 1, B);
            }
            mat_set_identity<T>(Id);

            i32 cond = (k >= 1);
            mat_axpy_beta<T, U>(1, B, 1, F);
            mat_axpy_beta<T, U>(1 - cond, Id, cond, I);

            mat_gemm<T, U>(1, F, I, 0, B);
            mat_copy<T>(B, F);
        }
        mat_plus_equal_scalar_id<T, U>(F, 1);
    }

    /**
     * @brief matrix scaling-squaring Talylor-based matrix exponential
     * @param K maximum number of matrix product allow
     * @param A input matrix
     * @param F output matrix
     * @param B,I,Id matrices
     * @param size_A size of matrices
     */
    template<
        typename T,
        class U,
        class Extents1,
        class Extents2,
        class Extents3,
        class Extents4,
        class Extents5,
        class Layout1,
        class Layout2,
        class Layout3,
        class Layout4,
        class Layout5,
        class Accessor1,
        class Accessor2,
        class Accessor3,
        class Accessor4,
        class Accessor5>
    inline void mat_exp(
        const i32 K,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &A,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &F,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &B,
        const std::mdspan<T, Extents4, Layout4, Accessor4> &I,
        const std::mdspan<T, Extents5, Layout5, Accessor5> &Id,
        const size_t size_A) {
        auto seq_mk        = sequence_mk();
        auto seq_qk        = sequence_qk();
        auto seq_rk        = sequence_rk();
        auto seq_ntheta_mk = sequence_nheta_mk();
        auto seq_bi        = define_bexp_coef();

        i32 k_star{0}, m_star{0}, s_star{0};
        // computation of k*, s*, m*
        order_scale<T>(K, seq_mk, seq_ntheta_mk, A, size_A, k_star, m_star, s_star);
        i32 r = seq_rk[k_star - 1];
        i32 q = seq_qk[k_star - 1];
        // scaling step
        i32 pw           = (1 << s_star);
        f64 scale_factor = 1.0 / pw;
        mat_mul_scalar<T>(A, scale_factor);

        // Taylor polynomial evaluation
        taylor_eval<T, U>(r, q, seq_bi, size_A, A, F, B, I, Id);

        // squaring step
        mat_set_identity<T>(Id);
        mat_set_identity<T>(I);

        for (auto j = 1; j <= pw; j++) {
            mat_copy<T>(I, Id);
            mat_gemm<T, U>(1, F, Id, 0, I);
        }
        mat_copy<T>(I, A);
    }
} // namespace shammath

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
 * @file offset_multipole.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shammath/symtensor_collections.hpp"

namespace shamphys {

    namespace details {

        template<class T>
        inline shammath::SymTensor3d_1<T> offset_multipole_1(
            const T &Qt0,
            const shammath::SymTensor3d_1<T> &Qt1,
            const shammath::SymTensor3d_1<T> &dt1) {
            return Qt1 + Qt0 * dt1;
        }

        template<class T>
        inline shammath::SymTensor3d_2<T> offset_multipole_2(
            const shammath::SymTensor3d_1<T> &Qn1,
            const shammath::SymTensor3d_1<T> &Qt1,
            const shammath::SymTensor3d_2<T> &Qt2,
            const shammath::SymTensor3d_1<T> &dt1) {
            // mathematica out
            // auto Qn2 = SymTensor3d_2<T>{
            //     Q.t2.v_00 + 2*Q.t1.v_0*d.t1.v_0 + Q.t0*d.t2.v_00,
            //     Q.t2.v_01 + Q.t1.v_1*d.t1.v_0 + Q.t1.v_0*d.t1.v_1 + Q.t0*d.t2.v_01,
            //     Q.t2.v_02 + Q.t1.v_2*d.t1.v_0 + Q.t1.v_0*d.t1.v_2 + Q.t0*d.t2.v_02,
            //     Q.t2.v_11 + 2*Q.t1.v_1*d.t1.v_1 + Q.t0*d.t2.v_11,
            //     Q.t2.v_12 + Q.t1.v_2*d.t1.v_1 + Q.t1.v_1*d.t1.v_2 + Q.t0*d.t2.v_12,
            //     Q.t2.v_22 + 2*Q.t1.v_2*d.t1.v_2 + Q.t0*d.t2.v_22
            // };

            // symbolic Qn2 : d_\mu Qn1_\nu + Q1_\mu d_\nu + Q2
            return shammath::SymTensor3d_2<T>{
                dt1.v_0 * Qn1.v_0 + Qt1.v_0 * dt1.v_0 + Qt2.v_00,
                dt1.v_0 * Qn1.v_1 + Qt1.v_0 * dt1.v_1 + Qt2.v_01,
                dt1.v_0 * Qn1.v_2 + Qt1.v_0 * dt1.v_2 + Qt2.v_02,
                dt1.v_1 * Qn1.v_1 + Qt1.v_1 * dt1.v_1 + Qt2.v_11,
                dt1.v_1 * Qn1.v_2 + Qt1.v_1 * dt1.v_2 + Qt2.v_12,
                dt1.v_2 * Qn1.v_2 + Qt1.v_2 * dt1.v_2 + Qt2.v_22};
        }

        template<class T>
        inline shammath::SymTensor3d_3<T> offset_multipole_3(
            const shammath::SymTensor3d_1<T> &Qn1,
            const shammath::SymTensor3d_2<T> &Qn2,
            const shammath::SymTensor3d_2<T> &Qt2,
            const shammath::SymTensor3d_3<T> &Qt3,
            const shammath::SymTensor3d_1<T> &dt1) {

            // mathematica out
            // auto Qn3 = SymTensor3d_3<T>{
            //     Qt3.v_000 + 3*Qt2.v_00*dt1.v_0 + 3*Q.t1.v_0*d.t2.v_00 + Q.t0*d.t3.v_000,
            //     Qt3.v_001 + 2*Qt2.v_01*dt1.v_0 + Qt2.v_00*dt1.v_1 + Q.t1.v_1*d.t2.v_00 +
            //     2*Q.t1.v_0*d.t2.v_01 + Q.t0*d.t3.v_001, Qt3.v_002 + 2*Qt2.v_02*dt1.v_0 +
            //     Qt2.v_00*dt1.v_2 + Q.t1.v_2*d.t2.v_00 + 2*Q.t1.v_0*d.t2.v_02 + Q.t0*d.t3.v_002,
            //     Qt3.v_011 + Qt2.v_11*dt1.v_0 + 2*Qt2.v_01*dt1.v_1 + 2*Q.t1.v_1*d.t2.v_01 +
            //     Q.t1.v_0*d.t2.v_11 + Q.t0*d.t3.v_011, Qt3.v_012 + Qt2.v_12*dt1.v_0 +
            //     Qt2.v_02*dt1.v_1 + Qt2.v_01*dt1.v_2 + Q.t1.v_2*d.t2.v_01 + Q.t1.v_1*d.t2.v_02
            //     + Q.t1.v_0*d.t2.v_12 + Q.t0*d.t3.v_012, Qt3.v_022 + Qt2.v_22*dt1.v_0 +
            //     2*Qt2.v_02*dt1.v_2 + 2*Q.t1.v_2*d.t2.v_02 + Q.t1.v_0*d.t2.v_22 +
            //     Q.t0*d.t3.v_022, Qt3.v_111 + 3*Qt2.v_11*dt1.v_1 + 3*Q.t1.v_1*d.t2.v_11 +
            //     Q.t0*d.t3.v_111, Qt3.v_112 + 2*Qt2.v_12*dt1.v_1 + Qt2.v_11*dt1.v_2 +
            //     Q.t1.v_2*d.t2.v_11 + 2*Q.t1.v_1*d.t2.v_12 + Q.t0*d.t3.v_112, Qt3.v_122 +
            //     Qt2.v_22*dt1.v_1 + 2*Qt2.v_12*dt1.v_2 + 2*Q.t1.v_2*d.t2.v_12 +
            //     Q.t1.v_1*d.t2.v_22 + Q.t0*d.t3.v_122, Qt3.v_222 + 3*Qt2.v_22*dt1.v_2 +
            //     3*Q.t1.v_2*d.t2.v_22 + Q.t0*d.t3.v_222
            // };

            // symbolic Qn3 : d_\mu Qn2_\nu\delta + d_\nu * (Qn2_ \mu \delta - d_\mu Qn1_\delta) +
            // Q2_\mu\nu d_\delta + Q3
            return shammath::SymTensor3d_3<T>{
                dt1.v_0 * Qn2.v_00 + dt1.v_0 * (Qn2.v_00 - dt1.v_0 * Qn1.v_0) + Qt2.v_00 * dt1.v_0
                    + Qt3.v_000,
                dt1.v_0 * Qn2.v_01 + dt1.v_0 * (Qn2.v_01 - dt1.v_0 * Qn1.v_1) + Qt2.v_00 * dt1.v_1
                    + Qt3.v_001,
                dt1.v_0 * Qn2.v_02 + dt1.v_0 * (Qn2.v_02 - dt1.v_0 * Qn1.v_2) + Qt2.v_00 * dt1.v_2
                    + Qt3.v_002,
                dt1.v_0 * Qn2.v_11 + dt1.v_1 * (Qn2.v_01 - dt1.v_0 * Qn1.v_1) + Qt2.v_01 * dt1.v_1
                    + Qt3.v_011,
                dt1.v_0 * Qn2.v_12 + dt1.v_1 * (Qn2.v_02 - dt1.v_0 * Qn1.v_2) + Qt2.v_01 * dt1.v_2
                    + Qt3.v_012,
                dt1.v_0 * Qn2.v_22 + dt1.v_2 * (Qn2.v_02 - dt1.v_0 * Qn1.v_2) + Qt2.v_02 * dt1.v_2
                    + Qt3.v_022,
                dt1.v_1 * Qn2.v_11 + dt1.v_1 * (Qn2.v_11 - dt1.v_1 * Qn1.v_1) + Qt2.v_11 * dt1.v_1
                    + Qt3.v_111,
                dt1.v_1 * Qn2.v_12 + dt1.v_1 * (Qn2.v_12 - dt1.v_1 * Qn1.v_2) + Qt2.v_11 * dt1.v_2
                    + Qt3.v_112,
                dt1.v_1 * Qn2.v_22 + dt1.v_2 * (Qn2.v_12 - dt1.v_1 * Qn1.v_2) + Qt2.v_12 * dt1.v_2
                    + Qt3.v_122,
                dt1.v_2 * Qn2.v_22 + dt1.v_2 * (Qn2.v_22 - dt1.v_2 * Qn1.v_2) + Qt2.v_22 * dt1.v_2
                    + Qt3.v_222};
        }

        template<class T>
        inline shammath::SymTensor3d_4<T> offset_multipole_4(
            const shammath::SymTensor3d_2<T> &Qn2,
            const shammath::SymTensor3d_3<T> &Qn3,
            const shammath::SymTensor3d_2<T> &Qt2,
            const shammath::SymTensor3d_3<T> &Qt3,
            const shammath::SymTensor3d_4<T> &Qt4,
            const shammath::SymTensor3d_1<T> &dt1,
            const shammath::SymTensor3d_2<T> &dt2) {

            // auto Qn4 = SymTensor3d_4<T>{
            //     Q.t4.v_0000 + 4*Q.t3.v_000*d.t1.v_0 + 6*Q.t2.v_00*d.t2.v_00 +
            //     4*Q.t1.v_0*d.t3.v_000 + Q.t0*d.t4.v_0000, Q.t4.v_0001 + 3*Q.t3.v_001*d.t1.v_0 +
            //     Q.t3.v_000*d.t1.v_1 + 3*Q.t2.v_01*d.t2.v_00 + 3*Q.t2.v_00*d.t2.v_01 +
            //     Q.t1.v_1*d.t3.v_000 + 3*Q.t1.v_0*d.t3.v_001 + Q.t0*d.t4.v_0001, Q.t4.v_0002 +
            //     3*Q.t3.v_002*d.t1.v_0 + Q.t3.v_000*d.t1.v_2 + 3*Q.t2.v_02*d.t2.v_00 +
            //     3*Q.t2.v_00*d.t2.v_02 + Q.t1.v_2*d.t3.v_000
            //     + 3*Q.t1.v_0*d.t3.v_002 + Q.t0*d.t4.v_0002, Q.t4.v_0011 + 2*Q.t3.v_011*d.t1.v_0 +
            //     2*Q.t3.v_001*d.t1.v_1 + Q.t2.v_11*d.t2.v_00 + 4*Q.t2.v_01*d.t2.v_01 +
            //     Q.t2.v_00*d.t2.v_11
            //     + 2*Q.t1.v_1*d.t3.v_001 + 2*Q.t1.v_0*d.t3.v_011 + Q.t0*d.t4.v_0011, Q.t4.v_0012 +
            //     2*Q.t3.v_012*d.t1.v_0 + Q.t3.v_002*d.t1.v_1 + Q.t3.v_001*d.t1.v_2 +
            //     Q.t2.v_12*d.t2.v_00 + 2*Q.t2.v_02*d.t2.v_01 + 2*Q.t2.v_01*d.t2.v_02 +
            //     Q.t2.v_00*d.t2.v_12 + Q.t1.v_2*d.t3.v_001
            //     + Q.t1.v_1*d.t3.v_002 + 2*Q.t1.v_0*d.t3.v_012 + Q.t0*d.t4.v_0012, Q.t4.v_0022 +
            //     2*Q.t3.v_022*d.t1.v_0 + 2*Q.t3.v_002*d.t1.v_2 + Q.t2.v_22*d.t2.v_00 +
            //     4*Q.t2.v_02*d.t2.v_02 + Q.t2.v_00*d.t2.v_22 + 2*Q.t1.v_2*d.t3.v_002 +
            //     2*Q.t1.v_0*d.t3.v_022 + Q.t0*d.t4.v_0022, Q.t4.v_0111 + Q.t3.v_111*d.t1.v_0 +
            //     3*Q.t3.v_011*d.t1.v_1 + 3*Q.t2.v_11*d.t2.v_01 + 3*Q.t2.v_01*d.t2.v_11 +
            //     3*Q.t1.v_1*d.t3.v_011 + Q.t1.v_0*d.t3.v_111 + Q.t0*d.t4.v_0111, Q.t4.v_0112 +
            //     Q.t3.v_112*d.t1.v_0 + 2*Q.t3.v_012*d.t1.v_1 + Q.t3.v_011*d.t1.v_2 +
            //     2*Q.t2.v_12*d.t2.v_01
            //     + Q.t2.v_11*d.t2.v_02 + Q.t2.v_02*d.t2.v_11 + 2*Q.t2.v_01*d.t2.v_12 +
            //     Q.t1.v_2*d.t3.v_011
            //     + 2*Q.t1.v_1*d.t3.v_012 + Q.t1.v_0*d.t3.v_112 + Q.t0*d.t4.v_0112, Q.t4.v_0122 +
            //     Q.t3.v_122*d.t1.v_0 + Q.t3.v_022*d.t1.v_1 + 2*Q.t3.v_012*d.t1.v_2 +
            //     Q.t2.v_22*d.t2.v_01 + 2*Q.t2.v_12*d.t2.v_02 + 2*Q.t2.v_02*d.t2.v_12 +
            //     Q.t2.v_01*d.t2.v_22 + 2*Q.t1.v_2*d.t3.v_012 + Q.t1.v_1*d.t3.v_022 +
            //     Q.t1.v_0*d.t3.v_122 + Q.t0*d.t4.v_0122, Q.t4.v_0222 + Q.t3.v_222*d.t1.v_0 +
            //     3*Q.t3.v_022*d.t1.v_2 + 3*Q.t2.v_22*d.t2.v_02 + 3*Q.t2.v_02*d.t2.v_22 +
            //     3*Q.t1.v_2*d.t3.v_022 + Q.t1.v_0*d.t3.v_222 + Q.t0*d.t4.v_0222, Q.t4.v_1111 +
            //     4*Q.t3.v_111*d.t1.v_1 + 6*Q.t2.v_11*d.t2.v_11 + 4*Q.t1.v_1*d.t3.v_111 +
            //     Q.t0*d.t4.v_1111, Q.t4.v_1112 + 3*Q.t3.v_112*d.t1.v_1 + Q.t3.v_111*d.t1.v_2 +
            //     3*Q.t2.v_12*d.t2.v_11 + 3*Q.t2.v_11*d.t2.v_12 + Q.t1.v_2*d.t3.v_111 +
            //     3*Q.t1.v_1*d.t3.v_112 + Q.t0*d.t4.v_1112, Q.t4.v_1122 + 2*Q.t3.v_122*d.t1.v_1 +
            //     2*Q.t3.v_112*d.t1.v_2 + Q.t2.v_22*d.t2.v_11 + 4*Q.t2.v_12*d.t2.v_12 +
            //     Q.t2.v_11*d.t2.v_22
            //     + 2*Q.t1.v_2*d.t3.v_112 + 2*Q.t1.v_1*d.t3.v_122 + Q.t0*d.t4.v_1122, Q.t4.v_1222 +
            //     Q.t3.v_222*d.t1.v_1 + 3*Q.t3.v_122*d.t1.v_2 + 3*Q.t2.v_22*d.t2.v_12 +
            //     3*Q.t2.v_12*d.t2.v_22 + 3*Q.t1.v_2*d.t3.v_122 + Q.t1.v_1*d.t3.v_222 +
            //     Q.t0*d.t4.v_1222, Q.t4.v_2222 + 4*Q.t3.v_222*d.t1.v_2 + 6*Q.t2.v_22*d.t2.v_22 +
            //     4*Q.t1.v_2*d.t3.v_222 + Q.t0*d.t4.v_2222
            // };

            // symbolic Qn4 : T4_\mu\nu\delta\epsilon = d_\mu * T3_\nu\delta\epsilon + d_\delta
            // *(T3_\mu\epsilon\nu - d_\mu * T2_\epsilon\nu ) + d2_\epsilon\nu *Q2_\delta\mu +
            // d_\epsilon *Q3_\nu\delta\mu  + d_\nu  *Q3_\delta\epsilon\mu + Q4_\mu\nu\delta\epsilon

            return shammath::SymTensor3d_4<T>{
                dt1.v_0 * Qn3.v_000 + dt1.v_0 * (Qn3.v_000 - dt1.v_0 * Qn2.v_00)
                    + dt2.v_00 * Qt2.v_00 + dt1.v_0 * Qt3.v_000 + dt1.v_0 * Qt3.v_000 + Qt4.v_0000,
                dt1.v_0 * Qn3.v_001 + dt1.v_0 * (Qn3.v_001 - dt1.v_0 * Qn2.v_01)
                    + dt2.v_01 * Qt2.v_00 + dt1.v_1 * Qt3.v_000 + dt1.v_0 * Qt3.v_001 + Qt4.v_0001,
                dt1.v_0 * Qn3.v_002 + dt1.v_0 * (Qn3.v_002 - dt1.v_0 * Qn2.v_02)
                    + dt2.v_02 * Qt2.v_00 + dt1.v_2 * Qt3.v_000 + dt1.v_0 * Qt3.v_002 + Qt4.v_0002,
                dt1.v_0 * Qn3.v_011 + dt1.v_1 * (Qn3.v_001 - dt1.v_0 * Qn2.v_01)
                    + dt2.v_01 * Qt2.v_01 + dt1.v_1 * Qt3.v_001 + dt1.v_0 * Qt3.v_011 + Qt4.v_0011,
                dt1.v_0 * Qn3.v_012 + dt1.v_1 * (Qn3.v_002 - dt1.v_0 * Qn2.v_02)
                    + dt2.v_02 * Qt2.v_01 + dt1.v_2 * Qt3.v_001 + dt1.v_0 * Qt3.v_012 + Qt4.v_0012,
                dt1.v_0 * Qn3.v_022 + dt1.v_2 * (Qn3.v_002 - dt1.v_0 * Qn2.v_02)
                    + dt2.v_02 * Qt2.v_02 + dt1.v_2 * Qt3.v_002 + dt1.v_0 * Qt3.v_022 + Qt4.v_0022,
                dt1.v_0 * Qn3.v_111 + dt1.v_1 * (Qn3.v_011 - dt1.v_0 * Qn2.v_11)
                    + dt2.v_11 * Qt2.v_01 + dt1.v_1 * Qt3.v_011 + dt1.v_1 * Qt3.v_011 + Qt4.v_0111,
                dt1.v_0 * Qn3.v_112 + dt1.v_1 * (Qn3.v_012 - dt1.v_0 * Qn2.v_12)
                    + dt2.v_12 * Qt2.v_01 + dt1.v_2 * Qt3.v_011 + dt1.v_1 * Qt3.v_012 + Qt4.v_0112,
                dt1.v_0 * Qn3.v_122 + dt1.v_2 * (Qn3.v_012 - dt1.v_0 * Qn2.v_12)
                    + dt2.v_12 * Qt2.v_02 + dt1.v_2 * Qt3.v_012 + dt1.v_1 * Qt3.v_022 + Qt4.v_0122,
                dt1.v_0 * Qn3.v_222 + dt1.v_2 * (Qn3.v_022 - dt1.v_0 * Qn2.v_22)
                    + dt2.v_22 * Qt2.v_02 + dt1.v_2 * Qt3.v_022 + dt1.v_2 * Qt3.v_022 + Qt4.v_0222,
                dt1.v_1 * Qn3.v_111 + dt1.v_1 * (Qn3.v_111 - dt1.v_1 * Qn2.v_11)
                    + dt2.v_11 * Qt2.v_11 + dt1.v_1 * Qt3.v_111 + dt1.v_1 * Qt3.v_111 + Qt4.v_1111,
                dt1.v_1 * Qn3.v_112 + dt1.v_1 * (Qn3.v_112 - dt1.v_1 * Qn2.v_12)
                    + dt2.v_12 * Qt2.v_11 + dt1.v_2 * Qt3.v_111 + dt1.v_1 * Qt3.v_112 + Qt4.v_1112,
                dt1.v_1 * Qn3.v_122 + dt1.v_2 * (Qn3.v_112 - dt1.v_1 * Qn2.v_12)
                    + dt2.v_12 * Qt2.v_12 + dt1.v_2 * Qt3.v_112 + dt1.v_1 * Qt3.v_122 + Qt4.v_1122,
                dt1.v_1 * Qn3.v_222 + dt1.v_2 * (Qn3.v_122 - dt1.v_1 * Qn2.v_22)
                    + dt2.v_22 * Qt2.v_12 + dt1.v_2 * Qt3.v_122 + dt1.v_2 * Qt3.v_122 + Qt4.v_1222,
                dt1.v_2 * Qn3.v_222 + dt1.v_2 * (Qn3.v_222 - dt1.v_2 * Qn2.v_22)
                    + dt2.v_22 * Qt2.v_22 + dt1.v_2 * Qt3.v_222 + dt1.v_2 * Qt3.v_222 + Qt4.v_2222};
        }

        template<class T>
        inline shammath::SymTensor3d_5<T> offset_multipole_5(
            const shammath::SymTensor3d_3<T> &Qn3,
            const shammath::SymTensor3d_4<T> &Qn4,
            const shammath::SymTensor3d_2<T> &Qt2,
            const shammath::SymTensor3d_3<T> &Qt3,
            const shammath::SymTensor3d_4<T> &Qt4,
            const shammath::SymTensor3d_5<T> &Qt5,
            const shammath::SymTensor3d_1<T> &dt1,
            const shammath::SymTensor3d_2<T> &dt2,
            const shammath::SymTensor3d_3<T> &dt3) {

            // symbolic Qn5 : {T5}_{\mu\nu\delta\epsilon\sigma} =  d_\mu  T4_\nu\delta\epsilon\sigma
            // + d_\nu ( T4_\mu\delta\epsilon\sigma  - d_\mu  T3_\delta\epsilon\sigma ) + Q2_\mu\nu
            // d3_\delta\epsilon\sigma +Q3_\mu\nu\sigma d2_\delta\epsilon +Q3_\mu\nu\delta
            // d2_\epsilon\sigma +Q4_\mu\nu\delta\sigma d_\epsilon +Q3_\mu\nu\epsilon
            // d2_\delta\sigma +Q4_\mu\nu\epsilon\sigma d_\delta +Q4_\mu\nu\delta\epsilon d_\sigma
            // +Q5_\mu\nu\delta\epsilon\sigma

            return shammath::SymTensor3d_5<T>{
                dt1.v_0 * Qn4.v_0000 + dt1.v_0 * (Qn4.v_0000 - dt1.v_0 * Qn3.v_000)
                    + Qt2.v_00 * dt3.v_000 + Qt3.v_000 * dt2.v_00 + Qt3.v_000 * dt2.v_00
                    + Qt4.v_0000 * dt1.v_0 + Qt3.v_000 * dt2.v_00 + Qt4.v_0000 * dt1.v_0
                    + Qt4.v_0000 * dt1.v_0 + Qt5.v_00000,
                dt1.v_0 * Qn4.v_0001 + dt1.v_0 * (Qn4.v_0001 - dt1.v_0 * Qn3.v_001)
                    + Qt2.v_00 * dt3.v_001 + Qt3.v_001 * dt2.v_00 + Qt3.v_000 * dt2.v_01
                    + Qt4.v_0001 * dt1.v_0 + Qt3.v_000 * dt2.v_01 + Qt4.v_0001 * dt1.v_0
                    + Qt4.v_0000 * dt1.v_1 + Qt5.v_00001,
                dt1.v_0 * Qn4.v_0002 + dt1.v_0 * (Qn4.v_0002 - dt1.v_0 * Qn3.v_002)
                    + Qt2.v_00 * dt3.v_002 + Qt3.v_002 * dt2.v_00 + Qt3.v_000 * dt2.v_02
                    + Qt4.v_0002 * dt1.v_0 + Qt3.v_000 * dt2.v_02 + Qt4.v_0002 * dt1.v_0
                    + Qt4.v_0000 * dt1.v_2 + Qt5.v_00002,
                dt1.v_0 * Qn4.v_0011 + dt1.v_0 * (Qn4.v_0011 - dt1.v_0 * Qn3.v_011)
                    + Qt2.v_00 * dt3.v_011 + Qt3.v_001 * dt2.v_01 + Qt3.v_000 * dt2.v_11
                    + Qt4.v_0001 * dt1.v_1 + Qt3.v_001 * dt2.v_01 + Qt4.v_0011 * dt1.v_0
                    + Qt4.v_0001 * dt1.v_1 + Qt5.v_00011,
                dt1.v_0 * Qn4.v_0012 + dt1.v_0 * (Qn4.v_0012 - dt1.v_0 * Qn3.v_012)
                    + Qt2.v_00 * dt3.v_012 + Qt3.v_002 * dt2.v_01 + Qt3.v_000 * dt2.v_12
                    + Qt4.v_0002 * dt1.v_1 + Qt3.v_001 * dt2.v_02 + Qt4.v_0012 * dt1.v_0
                    + Qt4.v_0001 * dt1.v_2 + Qt5.v_00012,
                dt1.v_0 * Qn4.v_0022 + dt1.v_0 * (Qn4.v_0022 - dt1.v_0 * Qn3.v_022)
                    + Qt2.v_00 * dt3.v_022 + Qt3.v_002 * dt2.v_02 + Qt3.v_000 * dt2.v_22
                    + Qt4.v_0002 * dt1.v_2 + Qt3.v_002 * dt2.v_02 + Qt4.v_0022 * dt1.v_0
                    + Qt4.v_0002 * dt1.v_2 + Qt5.v_00022,
                dt1.v_0 * Qn4.v_0111 + dt1.v_0 * (Qn4.v_0111 - dt1.v_0 * Qn3.v_111)
                    + Qt2.v_00 * dt3.v_111 + Qt3.v_001 * dt2.v_11 + Qt3.v_001 * dt2.v_11
                    + Qt4.v_0011 * dt1.v_1 + Qt3.v_001 * dt2.v_11 + Qt4.v_0011 * dt1.v_1
                    + Qt4.v_0011 * dt1.v_1 + Qt5.v_00111,
                dt1.v_0 * Qn4.v_0112 + dt1.v_0 * (Qn4.v_0112 - dt1.v_0 * Qn3.v_112)
                    + Qt2.v_00 * dt3.v_112 + Qt3.v_002 * dt2.v_11 + Qt3.v_001 * dt2.v_12
                    + Qt4.v_0012 * dt1.v_1 + Qt3.v_001 * dt2.v_12 + Qt4.v_0012 * dt1.v_1
                    + Qt4.v_0011 * dt1.v_2 + Qt5.v_00112,
                dt1.v_0 * Qn4.v_0122 + dt1.v_0 * (Qn4.v_0122 - dt1.v_0 * Qn3.v_122)
                    + Qt2.v_00 * dt3.v_122 + Qt3.v_002 * dt2.v_12 + Qt3.v_001 * dt2.v_22
                    + Qt4.v_0012 * dt1.v_2 + Qt3.v_002 * dt2.v_12 + Qt4.v_0022 * dt1.v_1
                    + Qt4.v_0012 * dt1.v_2 + Qt5.v_00122,
                dt1.v_0 * Qn4.v_0222 + dt1.v_0 * (Qn4.v_0222 - dt1.v_0 * Qn3.v_222)
                    + Qt2.v_00 * dt3.v_222 + Qt3.v_002 * dt2.v_22 + Qt3.v_002 * dt2.v_22
                    + Qt4.v_0022 * dt1.v_2 + Qt3.v_002 * dt2.v_22 + Qt4.v_0022 * dt1.v_2
                    + Qt4.v_0022 * dt1.v_2 + Qt5.v_00222,
                dt1.v_0 * Qn4.v_1111 + dt1.v_1 * (Qn4.v_0111 - dt1.v_0 * Qn3.v_111)
                    + Qt2.v_01 * dt3.v_111 + Qt3.v_011 * dt2.v_11 + Qt3.v_011 * dt2.v_11
                    + Qt4.v_0111 * dt1.v_1 + Qt3.v_011 * dt2.v_11 + Qt4.v_0111 * dt1.v_1
                    + Qt4.v_0111 * dt1.v_1 + Qt5.v_01111,
                dt1.v_0 * Qn4.v_1112 + dt1.v_1 * (Qn4.v_0112 - dt1.v_0 * Qn3.v_112)
                    + Qt2.v_01 * dt3.v_112 + Qt3.v_012 * dt2.v_11 + Qt3.v_011 * dt2.v_12
                    + Qt4.v_0112 * dt1.v_1 + Qt3.v_011 * dt2.v_12 + Qt4.v_0112 * dt1.v_1
                    + Qt4.v_0111 * dt1.v_2 + Qt5.v_01112,
                dt1.v_0 * Qn4.v_1122 + dt1.v_1 * (Qn4.v_0122 - dt1.v_0 * Qn3.v_122)
                    + Qt2.v_01 * dt3.v_122 + Qt3.v_012 * dt2.v_12 + Qt3.v_011 * dt2.v_22
                    + Qt4.v_0112 * dt1.v_2 + Qt3.v_012 * dt2.v_12 + Qt4.v_0122 * dt1.v_1
                    + Qt4.v_0112 * dt1.v_2 + Qt5.v_01122,
                dt1.v_0 * Qn4.v_1222 + dt1.v_1 * (Qn4.v_0222 - dt1.v_0 * Qn3.v_222)
                    + Qt2.v_01 * dt3.v_222 + Qt3.v_012 * dt2.v_22 + Qt3.v_012 * dt2.v_22
                    + Qt4.v_0122 * dt1.v_2 + Qt3.v_012 * dt2.v_22 + Qt4.v_0122 * dt1.v_2
                    + Qt4.v_0122 * dt1.v_2 + Qt5.v_01222,
                dt1.v_0 * Qn4.v_2222 + dt1.v_2 * (Qn4.v_0222 - dt1.v_0 * Qn3.v_222)
                    + Qt2.v_02 * dt3.v_222 + Qt3.v_022 * dt2.v_22 + Qt3.v_022 * dt2.v_22
                    + Qt4.v_0222 * dt1.v_2 + Qt3.v_022 * dt2.v_22 + Qt4.v_0222 * dt1.v_2
                    + Qt4.v_0222 * dt1.v_2 + Qt5.v_02222,
                dt1.v_1 * Qn4.v_1111 + dt1.v_1 * (Qn4.v_1111 - dt1.v_1 * Qn3.v_111)
                    + Qt2.v_11 * dt3.v_111 + Qt3.v_111 * dt2.v_11 + Qt3.v_111 * dt2.v_11
                    + Qt4.v_1111 * dt1.v_1 + Qt3.v_111 * dt2.v_11 + Qt4.v_1111 * dt1.v_1
                    + Qt4.v_1111 * dt1.v_1 + Qt5.v_11111,
                dt1.v_1 * Qn4.v_1112 + dt1.v_1 * (Qn4.v_1112 - dt1.v_1 * Qn3.v_112)
                    + Qt2.v_11 * dt3.v_112 + Qt3.v_112 * dt2.v_11 + Qt3.v_111 * dt2.v_12
                    + Qt4.v_1112 * dt1.v_1 + Qt3.v_111 * dt2.v_12 + Qt4.v_1112 * dt1.v_1
                    + Qt4.v_1111 * dt1.v_2 + Qt5.v_11112,
                dt1.v_1 * Qn4.v_1122 + dt1.v_1 * (Qn4.v_1122 - dt1.v_1 * Qn3.v_122)
                    + Qt2.v_11 * dt3.v_122 + Qt3.v_112 * dt2.v_12 + Qt3.v_111 * dt2.v_22
                    + Qt4.v_1112 * dt1.v_2 + Qt3.v_112 * dt2.v_12 + Qt4.v_1122 * dt1.v_1
                    + Qt4.v_1112 * dt1.v_2 + Qt5.v_11122,
                dt1.v_1 * Qn4.v_1222 + dt1.v_1 * (Qn4.v_1222 - dt1.v_1 * Qn3.v_222)
                    + Qt2.v_11 * dt3.v_222 + Qt3.v_112 * dt2.v_22 + Qt3.v_112 * dt2.v_22
                    + Qt4.v_1122 * dt1.v_2 + Qt3.v_112 * dt2.v_22 + Qt4.v_1122 * dt1.v_2
                    + Qt4.v_1122 * dt1.v_2 + Qt5.v_11222,
                dt1.v_1 * Qn4.v_2222 + dt1.v_2 * (Qn4.v_1222 - dt1.v_1 * Qn3.v_222)
                    + Qt2.v_12 * dt3.v_222 + Qt3.v_122 * dt2.v_22 + Qt3.v_122 * dt2.v_22
                    + Qt4.v_1222 * dt1.v_2 + Qt3.v_122 * dt2.v_22 + Qt4.v_1222 * dt1.v_2
                    + Qt4.v_1222 * dt1.v_2 + Qt5.v_12222,
                dt1.v_2 * Qn4.v_2222 + dt1.v_2 * (Qn4.v_2222 - dt1.v_2 * Qn3.v_222)
                    + Qt2.v_22 * dt3.v_222 + Qt3.v_222 * dt2.v_22 + Qt3.v_222 * dt2.v_22
                    + Qt4.v_2222 * dt1.v_2 + Qt3.v_222 * dt2.v_22 + Qt4.v_2222 * dt1.v_2
                    + Qt4.v_2222 * dt1.v_2 + Qt5.v_22222};

            // auto Qn5 = SymTensor3d_5<T>{
            //     Q.t5.v_00000 + 5*Q.t4.v_0000*d.t1.v_0 + 10*Q.t3.v_000*d.t2.v_00 +
            //     10*Q.t2.v_00*d.t3.v_000
            //     + 5*Q.t1.v_0*d.t4.v_0000 + Q.t0*d.t5.v_00000, Q.t5.v_00001 +
            //     4*Q.t4.v_0001*d.t1.v_0 + Q.t4.v_0000*d.t1.v_1 + 6*Q.t3.v_001*d.t2.v_00 +
            //     4*Q.t3.v_000*d.t2.v_01 + 4*Q.t2.v_01*d.t3.v_000 + 6*Q.t2.v_00*d.t3.v_001 +
            //     Q.t1.v_1*d.t4.v_0000 + 4*Q.t1.v_0*d.t4.v_0001 + Q.t0*d.t5.v_00001, Q.t5.v_00002 +
            //     4*Q.t4.v_0002*d.t1.v_0 + Q.t4.v_0000*d.t1.v_2 + 6*Q.t3.v_002*d.t2.v_00 +
            //     4*Q.t3.v_000*d.t2.v_02 + 4*Q.t2.v_02*d.t3.v_000 + 6*Q.t2.v_00*d.t3.v_002 +
            //     Q.t1.v_2*d.t4.v_0000 + 4*Q.t1.v_0*d.t4.v_0002 + Q.t0*d.t5.v_00002, Q.t5.v_00011 +
            //     3*Q.t4.v_0011*d.t1.v_0 + 2*Q.t4.v_0001*d.t1.v_1 + 3*Q.t3.v_011*d.t2.v_00 +
            //     6*Q.t3.v_001*d.t2.v_01 + Q.t3.v_000*d.t2.v_11 + Q.t2.v_11*d.t3.v_000 +
            //     6*Q.t2.v_01*d.t3.v_001 + 3*Q.t2.v_00*d.t3.v_011 + 2*Q.t1.v_1*d.t4.v_0001 +
            //     3*Q.t1.v_0*d.t4.v_0011 + Q.t0*d.t5.v_00011, Q.t5.v_00012 + 3*Q.t4.v_0012*d.t1.v_0
            //     + Q.t4.v_0002*d.t1.v_1 + Q.t4.v_0001*d.t1.v_2 + 3*Q.t3.v_012*d.t2.v_00 +
            //     3*Q.t3.v_002*d.t2.v_01 + 3*Q.t3.v_001*d.t2.v_02 + Q.t3.v_000*d.t2.v_12 +
            //     Q.t2.v_12*d.t3.v_000 + 3*Q.t2.v_02*d.t3.v_001 + 3*Q.t2.v_01*d.t3.v_002 +
            //     3*Q.t2.v_00*d.t3.v_012 + Q.t1.v_2*d.t4.v_0001 + Q.t1.v_1*d.t4.v_0002 +
            //     3*Q.t1.v_0*d.t4.v_0012 + Q.t0*d.t5.v_00012, Q.t5.v_00022 + 3*Q.t4.v_0022*d.t1.v_0
            //     + 2*Q.t4.v_0002*d.t1.v_2 + 3*Q.t3.v_022*d.t2.v_00 + 6*Q.t3.v_002*d.t2.v_02 +
            //     Q.t3.v_000*d.t2.v_22 + Q.t2.v_22*d.t3.v_000 + 6*Q.t2.v_02*d.t3.v_002 +
            //     3*Q.t2.v_00*d.t3.v_022 + 2*Q.t1.v_2*d.t4.v_0002 + 3*Q.t1.v_0*d.t4.v_0022 +
            //     Q.t0*d.t5.v_00022, Q.t5.v_00111 + 2*Q.t4.v_0111*d.t1.v_0 + 3*Q.t4.v_0011*d.t1.v_1
            //     + Q.t3.v_111*d.t2.v_00 + 6*Q.t3.v_011*d.t2.v_01 + 3*Q.t3.v_001*d.t2.v_11 +
            //     3*Q.t2.v_11*d.t3.v_001 + 6*Q.t2.v_01*d.t3.v_011 + Q.t2.v_00*d.t3.v_111 +
            //     3*Q.t1.v_1*d.t4.v_0011 + 2*Q.t1.v_0*d.t4.v_0111 + Q.t0*d.t5.v_00111, Q.t5.v_00112
            //     + 2*Q.t4.v_0112*d.t1.v_0 + 2*Q.t4.v_0012*d.t1.v_1 + Q.t4.v_0011*d.t1.v_2 +
            //     Q.t3.v_112*d.t2.v_00 + 4*Q.t3.v_012*d.t2.v_01 + 2*Q.t3.v_011*d.t2.v_02 +
            //     Q.t3.v_002*d.t2.v_11 + 2*Q.t3.v_001*d.t2.v_12 + 2*Q.t2.v_12*d.t3.v_001 +
            //     Q.t2.v_11*d.t3.v_002 + 2*Q.t2.v_02*d.t3.v_011 + 4*Q.t2.v_01*d.t3.v_012 +
            //     Q.t2.v_00*d.t3.v_112 + Q.t1.v_2*d.t4.v_0011 + 2*Q.t1.v_1*d.t4.v_0012 +
            //     2*Q.t1.v_0*d.t4.v_0112 + Q.t0*d.t5.v_00112, Q.t5.v_00122 + 2*Q.t4.v_0122*d.t1.v_0
            //     + Q.t4.v_0022*d.t1.v_1 + 2*Q.t4.v_0012*d.t1.v_2 + Q.t3.v_122*d.t2.v_00 +
            //     2*Q.t3.v_022*d.t2.v_01 + 4*Q.t3.v_012*d.t2.v_02 + 2*Q.t3.v_002*d.t2.v_12 +
            //     Q.t3.v_001*d.t2.v_22 + Q.t2.v_22*d.t3.v_001 + 2*Q.t2.v_12*d.t3.v_002 +
            //     4*Q.t2.v_02*d.t3.v_012 + 2*Q.t2.v_01*d.t3.v_022 + Q.t2.v_00*d.t3.v_122 +
            //     2*Q.t1.v_2*d.t4.v_0012 + Q.t1.v_1*d.t4.v_0022 + 2*Q.t1.v_0*d.t4.v_0122 +
            //     Q.t0*d.t5.v_00122, Q.t5.v_00222 + 2*Q.t4.v_0222*d.t1.v_0 + 3*Q.t4.v_0022*d.t1.v_2
            //     + Q.t3.v_222*d.t2.v_00 + 6*Q.t3.v_022*d.t2.v_02 + 3*Q.t3.v_002*d.t2.v_22 +
            //     3*Q.t2.v_22*d.t3.v_002 + 6*Q.t2.v_02*d.t3.v_022 + Q.t2.v_00*d.t3.v_222 +
            //     3*Q.t1.v_2*d.t4.v_0022 + 2*Q.t1.v_0*d.t4.v_0222 + Q.t0*d.t5.v_00222, Q.t5.v_01111
            //     + Q.t4.v_1111*d.t1.v_0 + 4*Q.t4.v_0111*d.t1.v_1 + 4*Q.t3.v_111*d.t2.v_01 +
            //     6*Q.t3.v_011*d.t2.v_11 + 6*Q.t2.v_11*d.t3.v_011 + 4*Q.t2.v_01*d.t3.v_111 +
            //     4*Q.t1.v_1*d.t4.v_0111 + Q.t1.v_0*d.t4.v_1111 + Q.t0*d.t5.v_01111, Q.t5.v_01112 +
            //     Q.t4.v_1112*d.t1.v_0 + 3*Q.t4.v_0112*d.t1.v_1 + Q.t4.v_0111*d.t1.v_2 +
            //     3*Q.t3.v_112*d.t2.v_01 + Q.t3.v_111*d.t2.v_02 + 3*Q.t3.v_012*d.t2.v_11 +
            //     3*Q.t3.v_011*d.t2.v_12 + 3*Q.t2.v_12*d.t3.v_011 + 3*Q.t2.v_11*d.t3.v_012 +
            //     Q.t2.v_02*d.t3.v_111 + 3*Q.t2.v_01*d.t3.v_112 + Q.t1.v_2*d.t4.v_0111 +
            //     3*Q.t1.v_1*d.t4.v_0112 + Q.t1.v_0*d.t4.v_1112 + Q.t0*d.t5.v_01112, Q.t5.v_01122 +
            //     Q.t4.v_1122*d.t1.v_0 + 2*Q.t4.v_0122*d.t1.v_1 + 2*Q.t4.v_0112*d.t1.v_2 +
            //     2*Q.t3.v_122*d.t2.v_01 + 2*Q.t3.v_112*d.t2.v_02 + Q.t3.v_022*d.t2.v_11 +
            //     4*Q.t3.v_012*d.t2.v_12 + Q.t3.v_011*d.t2.v_22 + Q.t2.v_22*d.t3.v_011 +
            //     4*Q.t2.v_12*d.t3.v_012 + Q.t2.v_11*d.t3.v_022 + 2*Q.t2.v_02*d.t3.v_112 +
            //     2*Q.t2.v_01*d.t3.v_122 + 2*Q.t1.v_2*d.t4.v_0112 + 2*Q.t1.v_1*d.t4.v_0122 +
            //     Q.t1.v_0*d.t4.v_1122 + Q.t0*d.t5.v_01122, Q.t5.v_01222 + Q.t4.v_1222*d.t1.v_0 +
            //     Q.t4.v_0222*d.t1.v_1 + 3*Q.t4.v_0122*d.t1.v_2 + Q.t3.v_222*d.t2.v_01 +
            //     3*Q.t3.v_122*d.t2.v_02 + 3*Q.t3.v_022*d.t2.v_12 + 3*Q.t3.v_012*d.t2.v_22 +
            //     3*Q.t2.v_22*d.t3.v_012 + 3*Q.t2.v_12*d.t3.v_022 + 3*Q.t2.v_02*d.t3.v_122 +
            //     Q.t2.v_01*d.t3.v_222 + 3*Q.t1.v_2*d.t4.v_0122 + Q.t1.v_1*d.t4.v_0222 +
            //     Q.t1.v_0*d.t4.v_1222 + Q.t0*d.t5.v_01222, Q.t5.v_02222 + Q.t4.v_2222*d.t1.v_0 +
            //     4*Q.t4.v_0222*d.t1.v_2 + 4*Q.t3.v_222*d.t2.v_02 + 6*Q.t3.v_022*d.t2.v_22 +
            //     6*Q.t2.v_22*d.t3.v_022 + 4*Q.t2.v_02*d.t3.v_222 + 4*Q.t1.v_2*d.t4.v_0222 +
            //     Q.t1.v_0*d.t4.v_2222 + Q.t0*d.t5.v_02222, Q.t5.v_11111 + 5*Q.t4.v_1111*d.t1.v_1 +
            //     10*Q.t3.v_111*d.t2.v_11 + 10*Q.t2.v_11*d.t3.v_111 + 5*Q.t1.v_1*d.t4.v_1111 +
            //     Q.t0*d.t5.v_11111, Q.t5.v_11112 + 4*Q.t4.v_1112*d.t1.v_1 + Q.t4.v_1111*d.t1.v_2 +
            //     6*Q.t3.v_112*d.t2.v_11 + 4*Q.t3.v_111*d.t2.v_12 + 4*Q.t2.v_12*d.t3.v_111 +
            //     6*Q.t2.v_11*d.t3.v_112 + Q.t1.v_2*d.t4.v_1111 + 4*Q.t1.v_1*d.t4.v_1112 +
            //     Q.t0*d.t5.v_11112, Q.t5.v_11122 + 3*Q.t4.v_1122*d.t1.v_1 + 2*Q.t4.v_1112*d.t1.v_2
            //     + 3*Q.t3.v_122*d.t2.v_11 + 6*Q.t3.v_112*d.t2.v_12 + Q.t3.v_111*d.t2.v_22 +
            //     Q.t2.v_22*d.t3.v_111 + 6*Q.t2.v_12*d.t3.v_112 + 3*Q.t2.v_11*d.t3.v_122 +
            //     2*Q.t1.v_2*d.t4.v_1112 + 3*Q.t1.v_1*d.t4.v_1122 + Q.t0*d.t5.v_11122, Q.t5.v_11222
            //     + 2*Q.t4.v_1222*d.t1.v_1 + 3*Q.t4.v_1122*d.t1.v_2 + Q.t3.v_222*d.t2.v_11 +
            //     6*Q.t3.v_122*d.t2.v_12 + 3*Q.t3.v_112*d.t2.v_22 + 3*Q.t2.v_22*d.t3.v_112 +
            //     6*Q.t2.v_12*d.t3.v_122 + Q.t2.v_11*d.t3.v_222 + 3*Q.t1.v_2*d.t4.v_1122 +
            //     2*Q.t1.v_1*d.t4.v_1222 + Q.t0*d.t5.v_11222, Q.t5.v_12222 + Q.t4.v_2222*d.t1.v_1 +
            //     4*Q.t4.v_1222*d.t1.v_2 + 4*Q.t3.v_222*d.t2.v_12 + 6*Q.t3.v_122*d.t2.v_22 +
            //     6*Q.t2.v_22*d.t3.v_122 + 4*Q.t2.v_12*d.t3.v_222 + 4*Q.t1.v_2*d.t4.v_1222 +
            //     Q.t1.v_1*d.t4.v_2222 + Q.t0*d.t5.v_12222, Q.t5.v_22222 + 5*Q.t4.v_2222*d.t1.v_2 +
            //     10*Q.t3.v_222*d.t2.v_22 + 10*Q.t2.v_22*d.t3.v_222 + 5*Q.t1.v_2*d.t4.v_2222 +
            //     Q.t0*d.t5.v_22222
            // };
        }

    } // namespace details

    /// utility to offset a multipole, see PHD
    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> offset_multipole_delta(
        const shammath::SymTensorCollection<T, low_order, high_order> &Q,
        const sycl::vec<T, 3> &offset) {

        using namespace shammath;
        using namespace shamphys::details;

        if constexpr (low_order == 0 && high_order == 5) {
            SymTensorCollection<T, 1, 3> d = SymTensorCollection<T, 1, 3>::from_vec(offset);

            auto Qn1 = offset_multipole_1(Q.t0, Q.t1, d.t1);

            auto Qn2 = offset_multipole_2(Qn1, Q.t1, Q.t2, d.t1);

            auto Qn3 = offset_multipole_3(Qn1, Qn2, Q.t2, Q.t3, d.t1);

            auto Qn4 = offset_multipole_4(Qn2, Qn3, Q.t2, Q.t3, Q.t4, d.t1, d.t2);

            auto Qn5 = offset_multipole_5(Qn3, Qn4, Q.t2, Q.t3, Q.t4, Q.t5, d.t1, d.t2, d.t3);

            return {Q.t0, Qn1, Qn2, Qn3, Qn4, Qn5};
        } else if constexpr (low_order == 0 && high_order == 4) {

            SymTensorCollection<T, 1, 2> d = SymTensorCollection<T, 1, 2>::from_vec(offset);

            auto Qn1 = offset_multipole_1(Q.t0, Q.t1, d.t1);

            auto Qn2 = offset_multipole_2(Qn1, Q.t1, Q.t2, d.t1);

            auto Qn3 = offset_multipole_3(Qn1, Qn2, Q.t2, Q.t3, d.t1);

            auto Qn4 = offset_multipole_4(Qn2, Qn3, Q.t2, Q.t3, Q.t4, d.t1, d.t2);

            return {Q.t0, Qn1, Qn2, Qn3, Qn4};
        } else if constexpr (low_order == 0 && high_order == 3) {

            SymTensorCollection<T, 1, 1> d = SymTensorCollection<T, 1, 1>::from_vec(offset);

            auto Qn1 = offset_multipole_1(Q.t0, Q.t1, d.t1);

            auto Qn2 = offset_multipole_2(Qn1, Q.t1, Q.t2, d.t1);

            auto Qn3 = offset_multipole_3(Qn1, Qn2, Q.t2, Q.t3, d.t1);

            return {Q.t0, Qn1, Qn2, Qn3};
        } else if constexpr (low_order == 0 && high_order == 2) {

            SymTensorCollection<T, 1, 1> d = SymTensorCollection<T, 1, 1>::from_vec(offset);

            auto Qn1 = offset_multipole_1(Q.t0, Q.t1, d.t1);

            auto Qn2 = offset_multipole_2(Qn1, Q.t1, Q.t2, d.t1);

            return {Q.t0, Qn1, Qn2};
        } else {
            static_assert(shambase::always_false_v<T>, "This combinaition of orders is not valid");
        }
    }

    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> offset_multipole(
        const shammath::SymTensorCollection<T, low_order, high_order> &Q_old,
        const sycl::vec<T, 3> &from,
        const sycl::vec<T, 3> &to) {
        return offset_multipole_delta(Q_old, from - to);
    }

} // namespace shamphys

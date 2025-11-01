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
 * @file grav_moments.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shammath/symtensor_collections.hpp"

namespace shamphys {

    /// Contraction of the green function derivatives (D_n) and the multipole moments (Q_n)
    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> get_M_mat(
        const shammath::SymTensorCollection<T, low_order, high_order> &D,
        const shammath::SymTensorCollection<T, low_order, high_order> &Q) {

        using namespace shammath;

        if constexpr (low_order == 0 && high_order == 5) {

            const T &TD0                = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;
            const SymTensor3d_4<T> &TD4 = D.t4;
            const SymTensor3d_5<T> &TD5 = D.t5;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;
            const SymTensor3d_3<T> &TQ3 = Q.t3;
            const SymTensor3d_4<T> &TQ4 = Q.t4;
            const SymTensor3d_5<T> &TQ5 = Q.t5;

            auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * T(1 / 2.)
                       + ((TD3 * TQ3)) * T(1 / 6.) + ((TD4 * TQ4)) * T(1 / 24.)
                       + ((TD5 * TQ5)) * T(1 / 120.);
            auto M_1 = T(-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * T(1 / 2.)
                       - ((TD4 * TQ3)) * T(1 / 6.) - ((TD5 * TQ4)) * T(1 / 24.);
            auto M_2
                = (TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2)) * T(1 / 2.) + ((TD5 * TQ3)) * T(1 / 6.);
            auto M_3 = T(-1.) * (TD3 * TQ0) - (TD4 * TQ1) - ((TD5 * TQ2)) * T(1 / 2.);
            auto M_4 = (TD4 * TQ0) + (TD5 * TQ1);
            auto M_5 = T(-1.) * (TD5 * TQ0);

            return SymTensorCollection<T, 0, 5>{M_0, M_1, M_2, M_3, M_4, M_5};
        } else if constexpr (low_order == 0 && high_order == 4) {

            const T &TD0                = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;
            const SymTensor3d_4<T> &TD4 = D.t4;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;
            const SymTensor3d_3<T> &TQ3 = Q.t3;
            const SymTensor3d_4<T> &TQ4 = Q.t4;

            auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * T(1 / 2.)
                       + ((TD3 * TQ3)) * T(1 / 6.) + ((TD4 * TQ4)) * T(1 / 24.);
            auto M_1 = T(-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * T(1 / 2.)
                       - ((TD4 * TQ3)) * T(1 / 6.);
            auto M_2 = (TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2)) * T(1 / 2.);
            auto M_3 = T(-1.) * (TD3 * TQ0) - (TD4 * TQ1);
            auto M_4 = (TD4 * TQ0);

            return SymTensorCollection<T, 0, 4>{M_0, M_1, M_2, M_3, M_4};
        } else if constexpr (low_order == 0 && high_order == 3) {

            const T &TD0                = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;
            const SymTensor3d_3<T> &TQ3 = Q.t3;

            auto M_0
                = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * T(1 / 2.) + ((TD3 * TQ3)) * T(1 / 6.);
            auto M_1 = T(-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * T(1 / 2.);
            auto M_2 = (TD2 * TQ0) + (TD3 * TQ1);
            auto M_3 = T(-1.) * (TD3 * TQ0);

            return SymTensorCollection<T, 0, 3>{M_0, M_1, M_2, M_3};
        } else if constexpr (low_order == 0 && high_order == 2) {

            const T &TD0                = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;

            auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * T(1 / 2.);
            auto M_1 = T(-1.) * (TD1 * TQ0) - (TD2 * TQ1);
            auto M_2 = TD2 * TQ0;

            return SymTensorCollection<T, 0, 2>{M_0, M_1, M_2};
        } else if constexpr (low_order == 0 && high_order == 1) {

            const T &TD0                = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;

            auto M_0 = (TD0 * TQ0) + (TD1 * TQ1);
            auto M_1 = T(-1.) * (TD1 * TQ0);

            return SymTensorCollection<T, 0, 1>{M_0, M_1};
        } else if constexpr (low_order == 0 && high_order == 0) {

            const T &TD0 = D.t0;

            const T &TQ0 = Q.t0;

            auto M_0 = TD0 * TQ0;

            return SymTensorCollection<T, 0, 0>{M_0};
        } else {
            static_assert(shambase::always_false_v<T>, "This combination of orders is not valid");
        }
    }

    /// Contraction of the green function derivatives (D_{n+1}) and the multipole moments (Q_n).
    /// This is also the nabla of the M_mat
    template<class T, u32 high_order>
    inline shammath::SymTensorCollection<T, 1, high_order + 1> get_dM_mat(
        const shammath::SymTensorCollection<T, 1, high_order + 1> &D,
        const shammath::SymTensorCollection<T, 0, high_order> &Q) {

        using namespace shammath;

        if constexpr (high_order == 4) {
            // T & TD0 = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;
            const SymTensor3d_4<T> &TD4 = D.t4;
            const SymTensor3d_5<T> &TD5 = D.t5;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;
            const SymTensor3d_3<T> &TQ3 = Q.t3;
            const SymTensor3d_4<T> &TQ4 = Q.t4;
            // SymTensor3d_5<T> & TQ5 = Q.t5;

            constexpr T _1i2  = T(1. / 2.);
            constexpr T _1i6  = T(1. / 6.);
            constexpr T _1i24 = T(1. / 24.);

            auto M_1 = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * _1i2 + (TD4 * TQ3) * _1i6
                       + (TD5 * TQ4) * _1i24;
            auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1 - (TD4 * TQ2) * _1i2 - (TD5 * TQ3) * _1i6;
            auto M_3 = TD3 * TQ0 + TD4 * TQ1 + (TD5 * TQ2) * _1i2;
            auto M_4 = (T(-1.) * (TD4 * TQ0)) - TD5 * TQ1;
            auto M_5 = TD5 * TQ0;

            return SymTensorCollection<T, 1, 5>{M_1, M_2, M_3, M_4, M_5};
        } else if constexpr (high_order == 3) {

            // T & TD0 = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;
            const SymTensor3d_4<T> &TD4 = D.t4;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;
            const SymTensor3d_3<T> &TQ3 = Q.t3;

            auto M_1 = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * T(1. / 2.) + (TD4 * TQ3) * T(1. / 6.);
            auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1 - (TD4 * TQ2) * T(1. / 2.);
            auto M_3 = TD3 * TQ0 + TD4 * TQ1;
            auto M_4 = T(-1.) * (TD4 * TQ0);

            return SymTensorCollection<T, 1, 4>{M_1, M_2, M_3, M_4};
        } else if constexpr (high_order == 2) {

            // T & TD0 = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;
            const SymTensor3d_3<T> &TD3 = D.t3;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;
            const SymTensor3d_2<T> &TQ2 = Q.t2;

            auto M_1 = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * (1. / 2.);
            auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1;
            auto M_3 = TD3 * TQ0;

            return SymTensorCollection<T, 1, 3>{M_1, M_2, M_3};
        } else if constexpr (high_order == 1) {

            // T & TD0 = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;
            const SymTensor3d_2<T> &TD2 = D.t2;

            const T &TQ0                = Q.t0;
            const SymTensor3d_1<T> &TQ1 = Q.t1;

            auto M_1 = TD1 * TQ0 + TD2 * TQ1;
            auto M_2 = T(-1.) * TD2 * TQ0;

            return SymTensorCollection<T, 1, 2>{M_1, M_2};
        } else if constexpr (high_order == 0) {
            // T & TD0 = D.t0;
            const SymTensor3d_1<T> &TD1 = D.t1;

            const T &TQ0 = Q.t0;

            auto M_1 = TD1 * TQ0;

            return SymTensorCollection<T, 1, 1>{M_1};
        } else {
            static_assert(shambase::always_false_v<T>, "This combination of orders is not valid");
        }
    }
} // namespace shamphys

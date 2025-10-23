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
#include "shammath/symtensor_collections.hpp"

namespace shamphys {

    /// Contraction of the green function derivatives (D_n) and the multipole moments (Q_n)
    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> get_M_mat(
        shammath::SymTensorCollection<T, low_order, high_order> &D,
        shammath::SymTensorCollection<T, low_order, high_order> &Q);

    /// Contraction of the green function derivatives (D_{n+1}) and the multipole moments (Q_n).
    /// This is also the nabla of the M_mat
    template<class T, u32 high_order>
    inline shammath::SymTensorCollection<T, 1, high_order + 1> get_dM_mat(
        shammath::SymTensorCollection<T, 1, high_order + 1> &D,
        shammath::SymTensorCollection<T, 0, high_order> &Q);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementations for all cases
    // -----------
    // Do not look if you donw want your eyes to bleed, this is very VERY ugly
    ////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 5> get_M_mat(
        shammath::SymTensorCollection<T, 0, 5> &D, shammath::SymTensorCollection<T, 0, 5> &Q) {

        using namespace shammath;

        T &TD0                = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;
        SymTensor3d_4<T> &TD4 = D.t4;
        SymTensor3d_5<T> &TD5 = D.t5;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;
        SymTensor3d_3<T> &TQ3 = Q.t3;
        SymTensor3d_4<T> &TQ4 = Q.t4;
        SymTensor3d_5<T> &TQ5 = Q.t5;

        auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * (1 / 2.) + ((TD3 * TQ3)) * (1 / 6.)
                   + ((TD4 * TQ4)) * (1 / 24.) + ((TD5 * TQ5)) * (1 / 120.);
        auto M_1 = (-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * (1 / 2.)
                   - ((TD4 * TQ3)) * (1 / 6.) - ((TD5 * TQ4)) * (1 / 24.);
        auto M_2
            = (1. / 2.)
              * ((TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2)) * (1 / 2.) + ((TD5 * TQ3)) * (1 / 6.));
        auto M_3 = (1. / 6.) * ((-1.) * (TD3 * TQ0) - (TD4 * TQ1) - ((TD5 * TQ2)) * (1 / 2.));
        auto M_4 = (1. / 24.) * ((TD4 * TQ0) + (TD5 * TQ1));
        auto M_5 = (-1. / 120) * (TD5 * TQ0);

        return SymTensorCollection<T, 0, 5>{M_0, M_1, M_2, M_3, M_4, M_5};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 4> get_M_mat(
        shammath::SymTensorCollection<T, 0, 4> &D, shammath::SymTensorCollection<T, 0, 4> &Q) {

        using namespace shammath;

        T &TD0                = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;
        SymTensor3d_4<T> &TD4 = D.t4;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;
        SymTensor3d_3<T> &TQ3 = Q.t3;
        SymTensor3d_4<T> &TQ4 = Q.t4;

        auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * (1 / 2.) + ((TD3 * TQ3)) * (1 / 6.)
                   + ((TD4 * TQ4)) * (1 / 24.);
        auto M_1 = (-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * (1 / 2.)
                   - ((TD4 * TQ3)) * (1 / 6.);
        auto M_2 = (1. / 2.) * ((TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2)) * (1 / 2.));
        auto M_3 = (1. / 6.) * ((-1.) * (TD3 * TQ0) - (TD4 * TQ1));
        auto M_4 = (1. / 24.) * ((TD4 * TQ0));

        return SymTensorCollection<T, 0, 4>{M_0, M_1, M_2, M_3, M_4};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 3> get_M_mat(
        shammath::SymTensorCollection<T, 0, 3> &D, shammath::SymTensorCollection<T, 0, 3> &Q) {

        using namespace shammath;

        T &TD0                = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;
        SymTensor3d_3<T> &TQ3 = Q.t3;

        auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * (1 / 2.) + ((TD3 * TQ3)) * (1 / 6.);
        auto M_1 = (-1.) * (TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2)) * (1 / 2.);
        auto M_2 = (1. / 2.) * ((TD2 * TQ0) + (TD3 * TQ1));
        auto M_3 = (1. / 6.) * ((-1.) * (TD3 * TQ0));

        return SymTensorCollection<T, 0, 3>{M_0, M_1, M_2, M_3};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 2> get_M_mat(
        shammath::SymTensorCollection<T, 0, 2> &D, shammath::SymTensorCollection<T, 0, 2> &Q) {

        using namespace shammath;

        T &TD0                = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;

        auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2)) * (1 / 2.);
        auto M_1 = (-1.) * (TD1 * TQ0) - (TD2 * TQ1);
        auto M_2 = (1. / 2.) * ((TD2 * TQ0));

        return SymTensorCollection<T, 0, 2>{M_0, M_1, M_2};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 1> get_M_mat(
        shammath::SymTensorCollection<T, 0, 1> &D, shammath::SymTensorCollection<T, 0, 1> &Q) {

        using namespace shammath;

        T &TD0                = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;

        auto M_0 = (TD0 * TQ0) + (TD1 * TQ1);
        auto M_1 = (-1.) * (TD1 * TQ0);

        return SymTensorCollection<T, 0, 1>{M_0, M_1};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 0, 0> get_M_mat(
        shammath::SymTensorCollection<T, 0, 0> &D, shammath::SymTensorCollection<T, 0, 0> &Q) {

        using namespace shammath;

        T &TD0 = D.t0;

        T &TQ0 = Q.t0;

        auto M_0 = (TD0 * TQ0);

        return SymTensorCollection<T, 0, 0>{M_0};
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline shammath::SymTensorCollection<T, 1, 5> get_dM_mat(
        shammath::SymTensorCollection<T, 1, 5> &D, shammath::SymTensorCollection<T, 0, 4> &Q) {

        using namespace shammath;
        // T & TD0 = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;
        SymTensor3d_4<T> &TD4 = D.t4;
        SymTensor3d_5<T> &TD5 = D.t5;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;
        SymTensor3d_3<T> &TQ3 = Q.t3;
        SymTensor3d_4<T> &TQ4 = Q.t4;
        // SymTensor3d_5<T> & TQ5 = Q.t5;

        constexpr T _1i2  = (1. / 2.);
        constexpr T _1i6  = (1. / 6.);
        constexpr T _1i24 = (1. / 24.);

        auto M_1
            = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * _1i2 + (TD4 * TQ3) * _1i6 + (TD5 * TQ4) * _1i24;
        auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1 - (TD4 * TQ2) * _1i2 - (TD5 * TQ3) * _1i6;
        auto M_3 = _1i2 * (TD3 * TQ0 + TD4 * TQ1 + (TD5 * TQ2) * _1i2);
        auto M_4 = _1i6 * ((T(-1.) * (TD4 * TQ0)) - TD5 * TQ1);
        auto M_5 = (TD5 * TQ0) * _1i24;

        return SymTensorCollection<T, 1, 5>{M_1, M_2, M_3, M_4, M_5};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 1, 4> get_dM_mat(
        shammath::SymTensorCollection<T, 1, 4> &D, shammath::SymTensorCollection<T, 0, 3> &Q) {

        using namespace shammath;

        // T & TD0 = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;
        SymTensor3d_4<T> &TD4 = D.t4;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;
        SymTensor3d_3<T> &TQ3 = Q.t3;

        auto M_1 = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * T(1. / 2.) + (TD4 * TQ3) * T(1. / 6.);
        auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1 - (TD4 * TQ2) * T(1. / 2.);
        auto M_3 = T(1. / 2.) * (TD3 * TQ0 + TD4 * TQ1);
        auto M_4 = T(1. / 6.) * ((T(-1.) * (TD4 * TQ0)));

        return SymTensorCollection<T, 1, 4>{M_1, M_2, M_3, M_4};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 1, 3> get_dM_mat(
        shammath::SymTensorCollection<T, 1, 3> &D, shammath::SymTensorCollection<T, 0, 2> &Q) {

        using namespace shammath;

        // T & TD0 = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;
        SymTensor3d_3<T> &TD3 = D.t3;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;
        SymTensor3d_2<T> &TQ2 = Q.t2;

        auto M_1 = TD1 * TQ0 + TD2 * TQ1 + (TD3 * TQ2) * (1. / 2.);
        auto M_2 = (T(-1.) * TD2 * TQ0) - TD3 * TQ1;
        auto M_3 = T(1. / 2.) * (TD3 * TQ0);

        return SymTensorCollection<T, 1, 3>{M_1, M_2, M_3};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 1, 2> get_dM_mat(
        shammath::SymTensorCollection<T, 1, 2> &D, shammath::SymTensorCollection<T, 0, 1> &Q) {

        using namespace shammath;

        // T & TD0 = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;
        SymTensor3d_2<T> &TD2 = D.t2;

        T &TQ0                = Q.t0;
        SymTensor3d_1<T> &TQ1 = Q.t1;

        auto M_1 = TD1 * TQ0 + TD2 * TQ1;
        auto M_2 = (T(-1.) * TD2 * TQ0);

        return SymTensorCollection<T, 1, 2>{M_1, M_2};
    }

    template<class T>
    inline shammath::SymTensorCollection<T, 1, 1> get_dM_mat(
        shammath::SymTensorCollection<T, 1, 1> &D, shammath::SymTensorCollection<T, 0, 0> &Q) {

        using namespace shammath;

        // T & TD0 = D.t0;
        SymTensor3d_1<T> &TD1 = D.t1;

        T &TQ0 = Q.t0;

        auto M_1 = TD1 * TQ0;

        return SymTensorCollection<T, 1, 1>{M_1};
    }

#endif
} // namespace shamphys

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
 * @file GreenFuncGravCartesian.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shammath/symtensor_collections.hpp"

namespace shamphys {

    /**
     * @brief Utility to get the derivatives of the Green function for gravity in Cartesian
     * coordinates
     *
     * At a given order i, the Green function derivative is denoted as \f$\nabla^{(i)}_r G(r)\f$
     * This function return the collection of the orders from low_order to high_order.
     *
     * @param T the type of the coordinates
     * @param low_order the lowest order of the derivative
     * @param high_order the highest order of the derivative
     * @return The symetric tensor collection of the derivatives of the Green function for gravity
     * in Cartesian coordinates
     */
    template<class T, u32 low_order, u32 high_order>
    class GreenFuncGravCartesian {
        public:
        inline static shammath::SymTensorCollection<T, low_order, high_order> get_der_tensors(
            sycl::vec<T, 3> r);
    };

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations for all cases
// -----------
// Do not look if you donw want your eyes to bleed, this is very VERY ugly
////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN
    template<class T>
    class GreenFuncGravCartesian<T, 0, 5> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 5> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T r1pow4 = r.x() * r1pow3;
            T r2pow4 = r.y() * r2pow3;
            T r3pow4 = r.z() * r3pow3;

            T r1pow5 = r.x() * r1pow4;
            T r2pow5 = r.y() * r2pow4;
            T r3pow5 = r.z() * r3pow4;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;
            T g4 = -7 * rm2 * g3;
            T g5 = -9 * rm2 * g4;

            auto D5 = SymTensor3d_5<T>{
                15 * g3 * r1 + 10 * g4 * r1pow3 + g5 * r1pow5,
                (3 * g3 + 6 * g4 * r1pow2 + g5 * r1pow4) * r2,
                (3 * g3 + 6 * g4 * r1pow2 + g5 * r1pow4) * r3,
                3 * g3 * r1 + g4 * r1pow3 + 3 * g4 * r1 * r2pow2 + g5 * r1pow3 * r2pow2,
                (3 * g4 * r1 + g5 * r1pow3) * r2 * r3,
                3 * g3 * r1 + g4 * r1pow3 + 3 * g4 * r1 * r3pow2 + g5 * r1pow3 * r3pow2,
                3 * g3 * r2 + 3 * g4 * r1pow2 * r2 + g4 * r2pow3 + g5 * r1pow2 * r2pow3,
                (g3 + g4 * r1pow2 + g4 * r2pow2 + g5 * r1pow2 * r2pow2) * r3,
                r2 * (g3 + g4 * r1pow2 + g4 * r3pow2 + g5 * r1pow2 * r3pow2),
                3 * g3 * r3 + 3 * g4 * r1pow2 * r3 + g4 * r3pow3 + g5 * r1pow2 * r3pow3,
                r1 * (3 * g3 + 6 * g4 * r2pow2 + g5 * r2pow4),
                r1 * (3 * g4 * r2 + g5 * r2pow3) * r3,
                r1 * (g3 + g4 * r2pow2 + g4 * r3pow2 + g5 * r2pow2 * r3pow2),
                r1 * r2 * (3 * g4 * r3 + g5 * r3pow3),
                r1 * (3 * g3 + 6 * g4 * r3pow2 + g5 * r3pow4),
                15 * g3 * r2 + 10 * g4 * r2pow3 + g5 * r2pow5,
                (3 * g3 + 6 * g4 * r2pow2 + g5 * r2pow4) * r3,
                3 * g3 * r2 + g4 * r2pow3 + 3 * g4 * r2 * r3pow2 + g5 * r2pow3 * r3pow2,
                3 * g3 * r3 + 3 * g4 * r2pow2 * r3 + g4 * r3pow3 + g5 * r2pow2 * r3pow3,
                r2 * (3 * g3 + 6 * g4 * r3pow2 + g5 * r3pow4),
                15 * g3 * r3 + 10 * g4 * r3pow3 + g5 * r3pow5};

            auto D4 = SymTensor3d_4<T>{
                3 * g2 + 6 * g3 * r1pow2 + g4 * r1pow4,
                3 * g3 * r1 * r2 + g4 * r1pow3 * r2,
                3 * g3 * r1 * r3 + g4 * r1pow3 * r3,
                g2 + g3 * r1pow2 + g3 * r2pow2 + g4 * r1pow2 * r2pow2,
                (g3 + g4 * r1pow2) * r2 * r3,
                g2 + g3 * r1pow2 + g3 * r3pow2 + g4 * r1pow2 * r3pow2,
                3 * g3 * r1 * r2 + g4 * r1 * r2pow3,
                r1 * (g3 + g4 * r2pow2) * r3,
                r1 * r2 * (g3 + g4 * r3pow2),
                3 * g3 * r1 * r3 + g4 * r1 * r3pow3,
                3 * g2 + 6 * g3 * r2pow2 + g4 * r2pow4,
                3 * g3 * r2 * r3 + g4 * r2pow3 * r3,
                g2 + g3 * r2pow2 + g3 * r3pow2 + g4 * r2pow2 * r3pow2,
                3 * g3 * r2 * r3 + g4 * r2 * r3pow3,
                3 * g2 + 6 * g3 * r3pow2 + g4 * r3pow4};

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            auto D0 = g0;

            return SymTensorCollection<T, 0, 5>{D0, D1, D2, D3, D4, D5};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 0, 4> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 4> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;
            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T r1pow4 = r.x() * r1pow3;
            T r2pow4 = r.y() * r2pow3;
            T r3pow4 = r.z() * r3pow3;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;
            T g4 = -7 * rm2 * g3;

            auto D4 = SymTensor3d_4<T>{
                3 * g2 + 6 * g3 * r1pow2 + g4 * r1pow4,
                3 * g3 * r1 * r2 + g4 * r1pow3 * r2,
                3 * g3 * r1 * r3 + g4 * r1pow3 * r3,
                g2 + g3 * r1pow2 + g3 * r2pow2 + g4 * r1pow2 * r2pow2,
                (g3 + g4 * r1pow2) * r2 * r3,
                g2 + g3 * r1pow2 + g3 * r3pow2 + g4 * r1pow2 * r3pow2,
                3 * g3 * r1 * r2 + g4 * r1 * r2pow3,
                r1 * (g3 + g4 * r2pow2) * r3,
                r1 * r2 * (g3 + g4 * r3pow2),
                3 * g3 * r1 * r3 + g4 * r1 * r3pow3,
                3 * g2 + 6 * g3 * r2pow2 + g4 * r2pow4,
                3 * g3 * r2 * r3 + g4 * r2pow3 * r3,
                g2 + g3 * r2pow2 + g3 * r3pow2 + g4 * r2pow2 * r3pow2,
                3 * g3 * r2 * r3 + g4 * r2 * r3pow3,
                3 * g2 + 6 * g3 * r3pow2 + g4 * r3pow4};

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            auto D0 = g0;

            return SymTensorCollection<T, 0, 4>{D0, D1, D2, D3, D4};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 0, 3> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 3> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            auto D0 = g0;

            return SymTensorCollection<T, 0, 3>{D0, D1, D2, D3};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 0, 2> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 2> get_der_tensors(sycl::vec<T, 3> r) {

            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            auto D0 = g0;

            return SymTensorCollection<T, 0, 2>{D0, D1, D2};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 0, 1> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 1> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;
            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            auto D0 = g0;

            return SymTensorCollection<T, 0, 1>{D0, D1};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 0, 0> {
        public:
        inline static shammath::SymTensorCollection<T, 0, 0> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;
            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;

            auto D0 = g0;

            return SymTensorCollection<T, 0, 0>{D0};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 1, 5> {
        public:
        inline static shammath::SymTensorCollection<T, 1, 5> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;
            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T r1pow4 = r.x() * r1pow3;
            T r2pow4 = r.y() * r2pow3;
            T r3pow4 = r.z() * r3pow3;

            T r1pow5 = r.x() * r1pow4;
            T r2pow5 = r.y() * r2pow4;
            T r3pow5 = r.z() * r3pow4;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;
            T g4 = -7 * rm2 * g3;
            T g5 = -9 * rm2 * g4;

            auto D5 = SymTensor3d_5<T>{
                15 * g3 * r1 + 10 * g4 * r1pow3 + g5 * r1pow5,
                (3 * g3 + 6 * g4 * r1pow2 + g5 * r1pow4) * r2,
                (3 * g3 + 6 * g4 * r1pow2 + g5 * r1pow4) * r3,
                3 * g3 * r1 + g4 * r1pow3 + 3 * g4 * r1 * r2pow2 + g5 * r1pow3 * r2pow2,
                (3 * g4 * r1 + g5 * r1pow3) * r2 * r3,
                3 * g3 * r1 + g4 * r1pow3 + 3 * g4 * r1 * r3pow2 + g5 * r1pow3 * r3pow2,
                3 * g3 * r2 + 3 * g4 * r1pow2 * r2 + g4 * r2pow3 + g5 * r1pow2 * r2pow3,
                (g3 + g4 * r1pow2 + g4 * r2pow2 + g5 * r1pow2 * r2pow2) * r3,
                r2 * (g3 + g4 * r1pow2 + g4 * r3pow2 + g5 * r1pow2 * r3pow2),
                3 * g3 * r3 + 3 * g4 * r1pow2 * r3 + g4 * r3pow3 + g5 * r1pow2 * r3pow3,
                r1 * (3 * g3 + 6 * g4 * r2pow2 + g5 * r2pow4),
                r1 * (3 * g4 * r2 + g5 * r2pow3) * r3,
                r1 * (g3 + g4 * r2pow2 + g4 * r3pow2 + g5 * r2pow2 * r3pow2),
                r1 * r2 * (3 * g4 * r3 + g5 * r3pow3),
                r1 * (3 * g3 + 6 * g4 * r3pow2 + g5 * r3pow4),
                15 * g3 * r2 + 10 * g4 * r2pow3 + g5 * r2pow5,
                (3 * g3 + 6 * g4 * r2pow2 + g5 * r2pow4) * r3,
                3 * g3 * r2 + g4 * r2pow3 + 3 * g4 * r2 * r3pow2 + g5 * r2pow3 * r3pow2,
                3 * g3 * r3 + 3 * g4 * r2pow2 * r3 + g4 * r3pow3 + g5 * r2pow2 * r3pow3,
                r2 * (3 * g3 + 6 * g4 * r3pow2 + g5 * r3pow4),
                15 * g3 * r3 + 10 * g4 * r3pow3 + g5 * r3pow5};

            auto D4 = SymTensor3d_4<T>{
                3 * g2 + 6 * g3 * r1pow2 + g4 * r1pow4,
                3 * g3 * r1 * r2 + g4 * r1pow3 * r2,
                3 * g3 * r1 * r3 + g4 * r1pow3 * r3,
                g2 + g3 * r1pow2 + g3 * r2pow2 + g4 * r1pow2 * r2pow2,
                (g3 + g4 * r1pow2) * r2 * r3,
                g2 + g3 * r1pow2 + g3 * r3pow2 + g4 * r1pow2 * r3pow2,
                3 * g3 * r1 * r2 + g4 * r1 * r2pow3,
                r1 * (g3 + g4 * r2pow2) * r3,
                r1 * r2 * (g3 + g4 * r3pow2),
                3 * g3 * r1 * r3 + g4 * r1 * r3pow3,
                3 * g2 + 6 * g3 * r2pow2 + g4 * r2pow4,
                3 * g3 * r2 * r3 + g4 * r2pow3 * r3,
                g2 + g3 * r2pow2 + g3 * r3pow2 + g4 * r2pow2 * r3pow2,
                3 * g3 * r2 * r3 + g4 * r2 * r3pow3,
                3 * g2 + 6 * g3 * r3pow2 + g4 * r3pow4};

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            return SymTensorCollection<T, 1, 5>{D1, D2, D3, D4, D5};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 1, 4> {
        public:
        inline static shammath::SymTensorCollection<T, 1, 4> get_der_tensors(sycl::vec<T, 3> r) {

            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T r1pow4 = r.x() * r1pow3;
            T r2pow4 = r.y() * r2pow3;
            T r3pow4 = r.z() * r3pow3;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;
            T g4 = -7 * rm2 * g3;

            auto D4 = SymTensor3d_4<T>{
                3 * g2 + 6 * g3 * r1pow2 + g4 * r1pow4,
                3 * g3 * r1 * r2 + g4 * r1pow3 * r2,
                3 * g3 * r1 * r3 + g4 * r1pow3 * r3,
                g2 + g3 * r1pow2 + g3 * r2pow2 + g4 * r1pow2 * r2pow2,
                (g3 + g4 * r1pow2) * r2 * r3,
                g2 + g3 * r1pow2 + g3 * r3pow2 + g4 * r1pow2 * r3pow2,
                3 * g3 * r1 * r2 + g4 * r1 * r2pow3,
                r1 * (g3 + g4 * r2pow2) * r3,
                r1 * r2 * (g3 + g4 * r3pow2),
                3 * g3 * r1 * r3 + g4 * r1 * r3pow3,
                3 * g2 + 6 * g3 * r2pow2 + g4 * r2pow4,
                3 * g3 * r2 * r3 + g4 * r2pow3 * r3,
                g2 + g3 * r2pow2 + g3 * r3pow2 + g4 * r2pow2 * r3pow2,
                3 * g3 * r2 * r3 + g4 * r2 * r3pow3,
                3 * g2 + 6 * g3 * r3pow2 + g4 * r3pow4};

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            return SymTensorCollection<T, 1, 4>{D1, D2, D3, D4};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 1, 3> {
        public:
        inline static shammath::SymTensorCollection<T, 1, 3> get_der_tensors(sycl::vec<T, 3> r) {

            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;
            T g3 = -5 * rm2 * g2;

            auto D3 = SymTensor3d_3<T>{
                3 * g2 * r1 + g3 * r1pow3,
                g2 * r2 + g3 * r1pow2 * r2,
                g2 * r3 + g3 * r1pow2 * r3,
                g2 * r1 + g3 * r1 * r2pow2,
                g3 * r1 * r2 * r3,
                g2 * r1 + g3 * r1 * r3pow2,
                3 * g2 * r2 + g3 * r2pow3,
                g2 * r3 + g3 * r2pow2 * r3,
                g2 * r2 + g3 * r2 * r3pow2,
                3 * g2 * r3 + g3 * r3pow3};

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            return SymTensorCollection<T, 1, 3>{D1, D2, D3};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 1, 2> {
        public:
        inline static shammath::SymTensorCollection<T, 1, 2> get_der_tensors(sycl::vec<T, 3> r) {

            using namespace shammath;

            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;
            T g2 = -3 * rm2 * g1;

            auto D2 = SymTensor3d_2<T>{
                g1 + g2 * r1pow2,
                g2 * r1 * r2,
                g2 * r1 * r3,
                g1 + g2 * r2pow2,
                g2 * r2 * r3,
                g1 + g2 * r3pow2};

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            return SymTensorCollection<T, 1, 2>{D1, D2};
        }
    };

    template<class T>
    class GreenFuncGravCartesian<T, 1, 1> {
        public:
        inline static shammath::SymTensorCollection<T, 1, 1> get_der_tensors(sycl::vec<T, 3> r) {
            using namespace shammath;
            T r1 = r.x();
            T r2 = r.y();
            T r3 = r.z();

            T r1pow2 = r.x() * r.x();
            T r2pow2 = r.y() * r.y();
            T r3pow2 = r.z() * r.z();

            T r1pow3 = r.x() * r1pow2;
            T r2pow3 = r.y() * r2pow2;
            T r3pow3 = r.z() * r3pow2;

            T rsq = r1pow2 + r2pow2 + r3pow2;

            T rnorm = sycl::sqrt(rsq);

            T rm2 = 1 / (rsq);

            T g0 = 1 / rnorm;
            T g1 = -1 * rm2 * g0;

            auto D1 = SymTensor3d_1<T>{g1 * r1, g1 * r2, g1 * r3};

            return SymTensorCollection<T, 1, 1>{D1};
        }
    };

#endif
} // namespace shamphys

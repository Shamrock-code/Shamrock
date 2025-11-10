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
 * @file symtensor_collections.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 * \todo move to math
 */

#include "shambackends/sycl.hpp"
#include "symtensors.hpp"

namespace shammath {

    template<class T, u32 low_order, u32 high_order>
    struct SymTensorCollection;

    template<class T>
    struct SymTensorCollection<T, 0, 5> {
        T t0;
        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;
        SymTensor3d_4<T> t4;
        SymTensor3d_5<T> t5;

        static constexpr u32 num_component
            = 1 + SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt
              + SymTensor3d_3<T>::compo_cnt + SymTensor3d_4<T>::compo_cnt
              + SymTensor3d_5<T>::compo_cnt;

        static constexpr u32 offset_t0 = 0;
        static constexpr u32 offset_t1 = 1;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;
        static constexpr u32 offset_t4 = offset_t3 + SymTensor3d_3<T>::compo_cnt;
        static constexpr u32 offset_t5 = offset_t4 + SymTensor3d_4<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            auto A4 = SymTensor3d_4<T>{
                A3.v_000 * A1.v_0,
                A3.v_001 * A1.v_0,
                A3.v_002 * A1.v_0,
                A3.v_011 * A1.v_0,
                A3.v_012 * A1.v_0,
                A3.v_022 * A1.v_0,
                A3.v_111 * A1.v_0,
                A3.v_112 * A1.v_0,
                A3.v_122 * A1.v_0,
                A3.v_222 * A1.v_0,
                A3.v_111 * A1.v_1,
                A3.v_112 * A1.v_1,
                A3.v_122 * A1.v_1,
                A3.v_222 * A1.v_1,
                A3.v_222 * A1.v_2};

            auto A5 = SymTensor3d_5<T>{A4.v_0000 * A1.v_0, A4.v_0001 * A1.v_0, A4.v_0002 * A1.v_0,
                                       A4.v_0011 * A1.v_0, A4.v_0012 * A1.v_0, A4.v_0022 * A1.v_0,
                                       A4.v_0111 * A1.v_0, A4.v_0112 * A1.v_0, A4.v_0122 * A1.v_0,
                                       A4.v_0222 * A1.v_0, A4.v_1111 * A1.v_0, A4.v_1112 * A1.v_0,
                                       A4.v_1122 * A1.v_0, A4.v_1222 * A1.v_0, A4.v_2222 * A1.v_0,
                                       A4.v_1111 * A1.v_1, A4.v_1112 * A1.v_1, A4.v_1122 * A1.v_1,
                                       A4.v_1222 * A1.v_1, A4.v_2222 * A1.v_1, A4.v_2222 * A1.v_2};

            return {1, A1, A2, A3, A4, A5};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A4 = SymTensor3d_4<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A5
                = SymTensor3d_5<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {0, A1, A2, A3, A4};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + offset_t0] = t0;
            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
            t4.store(acc, offset + offset_t4);
            t5.store(acc, offset + offset_t5);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                acc[offset + offset_t0],
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3),
                SymTensor3d_4<T>::load(acc, offset + offset_t4),
                SymTensor3d_5<T>::load(acc, offset + offset_t5)};
        }

        inline SymTensorCollection<T, 0, 5> &operator*=(const T scal) {

            t0 *= scal;
            t1 *= scal;
            t2 *= scal;
            t3 *= scal;
            t4 *= scal;
            t5 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 5> &operator+=(const SymTensorCollection<T, 0, 5> other) {

            t0 += other.t0;
            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;
            t4 += other.t4;
            t5 += other.t5;

            return *this;
        }

        inline SymTensorCollection<T, 0, 5> operator-(
            const SymTensorCollection<T, 0, 5> &other) const {
            return {
                t0 - other.t0,
                t1 - other.t1,
                t2 - other.t2,
                t3 - other.t3,
                t4 - other.t4,
                t5 - other.t5};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 0, 4> {
        T t0;
        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;
        SymTensor3d_4<T> t4;

        static constexpr u32 num_component
            = 1 + SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt
              + SymTensor3d_3<T>::compo_cnt + SymTensor3d_4<T>::compo_cnt;

        static constexpr u32 offset_t0 = 0;
        static constexpr u32 offset_t1 = 1;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;
        static constexpr u32 offset_t4 = offset_t3 + SymTensor3d_3<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            auto A4 = SymTensor3d_4<T>{
                A3.v_000 * A1.v_0,
                A3.v_001 * A1.v_0,
                A3.v_002 * A1.v_0,
                A3.v_011 * A1.v_0,
                A3.v_012 * A1.v_0,
                A3.v_022 * A1.v_0,
                A3.v_111 * A1.v_0,
                A3.v_112 * A1.v_0,
                A3.v_122 * A1.v_0,
                A3.v_222 * A1.v_0,
                A3.v_111 * A1.v_1,
                A3.v_112 * A1.v_1,
                A3.v_122 * A1.v_1,
                A3.v_222 * A1.v_1,
                A3.v_222 * A1.v_2};

            return {1, A1, A2, A3, A4};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A4 = SymTensor3d_4<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {0, A1, A2, A3, A4};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + offset_t0] = t0;
            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
            t4.store(acc, offset + offset_t4);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                acc[offset + offset_t0],
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3),
                SymTensor3d_4<T>::load(acc, offset + offset_t4)};
        }

        inline SymTensorCollection<T, 0, 4> &operator*=(const T scal) {

            t0 *= scal;
            t1 *= scal;
            t2 *= scal;
            t3 *= scal;
            t4 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 4> &operator+=(const SymTensorCollection<T, 0, 4> other) {

            t0 += other.t0;
            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;
            t4 += other.t4;

            return *this;
        }

        inline SymTensorCollection<T, 0, 4> operator-(
            const SymTensorCollection<T, 0, 4> &other) const {
            return {t0 - other.t0, t1 - other.t1, t2 - other.t2, t3 - other.t3, t4 - other.t4};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 0, 3> {
        T t0;
        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;

        static constexpr u32 num_component = 1 + SymTensor3d_1<T>::compo_cnt
                                             + SymTensor3d_2<T>::compo_cnt
                                             + SymTensor3d_3<T>::compo_cnt;

        static constexpr u32 offset_t0 = 0;
        static constexpr u32 offset_t1 = 1;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            return {1, A1, A2, A3};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {0, A1, A2, A3};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + offset_t0] = t0;
            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                acc[offset + offset_t0],
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3)};
        }

        inline SymTensorCollection<T, 0, 3> &operator*=(const T scal) {

            t0 *= scal;
            t1 *= scal;
            t2 *= scal;
            t3 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 3> &operator+=(const SymTensorCollection<T, 0, 3> other) {

            t0 += other.t0;
            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;

            return *this;
        }

        inline SymTensorCollection<T, 0, 3> operator-(
            const SymTensorCollection<T, 0, 3> &other) const {
            return {t0 - other.t0, t1 - other.t1, t2 - other.t2, t3 - other.t3};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 0, 2> {
        T t0;
        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;

        static constexpr u32 num_component
            = 1 + SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt;

        static constexpr u32 offset_t0 = 0;
        static constexpr u32 offset_t1 = 1;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            return {1, A1, A2};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            return {0, A1, A2};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + offset_t0] = t0;
            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                acc[offset + offset_t0],
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2)};
        }

        inline SymTensorCollection<T, 0, 2> &operator*=(const T scal) {

            t0 *= scal;
            t1 *= scal;
            t2 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 2> &operator+=(const SymTensorCollection<T, 0, 2> other) {

            t0 += other.t0;
            t1 += other.t1;
            t2 += other.t2;

            return *this;
        }

        inline SymTensorCollection<T, 0, 2> operator-(
            const SymTensorCollection<T, 0, 2> &other) const {
            return {t0 - other.t0, t1 - other.t1, t2 - other.t2};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 0, 1> {
        T t0;
        SymTensor3d_1<T> t1;

        static constexpr u32 num_component = 1 + SymTensor3d_1<T>::compo_cnt;

        static constexpr u32 offset_t0 = 0;
        static constexpr u32 offset_t1 = 1;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            return {1, A1};
        }
        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            return {0, A1};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + offset_t0] = t0;
            t1.store(acc, offset + offset_t1);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                acc[offset + offset_t0], SymTensor3d_1<T>::load(acc, offset + offset_t1)};
        }

        inline SymTensorCollection<T, 0, 1> &operator*=(const T scal) {

            t0 *= scal;
            t1 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 1> &operator+=(const SymTensorCollection<T, 0, 1> other) {

            t0 += other.t0;
            t1 += other.t1;

            return *this;
        }

        inline SymTensorCollection<T, 0, 1> operator-(
            const SymTensorCollection<T, 0, 1> &other) const {
            return {t0 - other.t0, t1 - other.t1};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 0, 0> {
        T t0;

        static constexpr u32 num_component = 1;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) { return {1}; }
        inline static SymTensorCollection zeros() { return {0}; }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {
            acc[offset + 0] = t0;
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{acc[offset + 0]};
        }

        inline SymTensorCollection<T, 0, 0> &operator*=(const T scal) {

            t0 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 0, 0> &operator+=(const SymTensorCollection<T, 0, 0> other) {

            t0 += other.t0;

            return *this;
        }

        inline SymTensorCollection<T, 0, 0> operator-(
            const SymTensorCollection<T, 0, 0> &other) const {
            return {t0 - other.t0};
        }
    };

    ////////////// starting order = 1

    template<class T>
    struct SymTensorCollection<T, 1, 5> {

        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;
        SymTensor3d_4<T> t4;
        SymTensor3d_5<T> t5;

        static constexpr u32 num_component
            = SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt
              + SymTensor3d_3<T>::compo_cnt + SymTensor3d_4<T>::compo_cnt
              + SymTensor3d_5<T>::compo_cnt;

        static constexpr u32 offset_t1 = 0;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;
        static constexpr u32 offset_t4 = offset_t3 + SymTensor3d_3<T>::compo_cnt;
        static constexpr u32 offset_t5 = offset_t4 + SymTensor3d_4<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            auto A4 = SymTensor3d_4<T>{
                A3.v_000 * A1.v_0,
                A3.v_001 * A1.v_0,
                A3.v_002 * A1.v_0,
                A3.v_011 * A1.v_0,
                A3.v_012 * A1.v_0,
                A3.v_022 * A1.v_0,
                A3.v_111 * A1.v_0,
                A3.v_112 * A1.v_0,
                A3.v_122 * A1.v_0,
                A3.v_222 * A1.v_0,
                A3.v_111 * A1.v_1,
                A3.v_112 * A1.v_1,
                A3.v_122 * A1.v_1,
                A3.v_222 * A1.v_1,
                A3.v_222 * A1.v_2};

            auto A5 = SymTensor3d_5<T>{A4.v_0000 * A1.v_0, A4.v_0001 * A1.v_0, A4.v_0002 * A1.v_0,
                                       A4.v_0011 * A1.v_0, A4.v_0012 * A1.v_0, A4.v_0022 * A1.v_0,
                                       A4.v_0111 * A1.v_0, A4.v_0112 * A1.v_0, A4.v_0122 * A1.v_0,
                                       A4.v_0222 * A1.v_0, A4.v_1111 * A1.v_0, A4.v_1112 * A1.v_0,
                                       A4.v_1122 * A1.v_0, A4.v_1222 * A1.v_0, A4.v_2222 * A1.v_0,
                                       A4.v_1111 * A1.v_1, A4.v_1112 * A1.v_1, A4.v_1122 * A1.v_1,
                                       A4.v_1222 * A1.v_1, A4.v_2222 * A1.v_1, A4.v_2222 * A1.v_2};

            return {A1, A2, A3, A4, A5};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A4 = SymTensor3d_4<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A5
                = SymTensor3d_5<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {A1, A2, A3, A4};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {

            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
            t4.store(acc, offset + offset_t4);
            t5.store(acc, offset + offset_t5);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3),
                SymTensor3d_4<T>::load(acc, offset + offset_t4),
                SymTensor3d_5<T>::load(acc, offset + offset_t5)};
        }

        inline SymTensorCollection<T, 1, 5> &operator*=(const T scal) {

            t1 *= scal;
            t2 *= scal;
            t3 *= scal;
            t4 *= scal;
            t5 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 1, 5> &operator+=(const SymTensorCollection<T, 1, 5> other) {

            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;
            t4 += other.t4;
            t5 += other.t5;

            return *this;
        }

        inline SymTensorCollection<T, 1, 5> operator-(
            const SymTensorCollection<T, 1, 5> &other) const {
            return {t1 - other.t1, t2 - other.t2, t3 - other.t3, t4 - other.t4, t5 - other.t5};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 1, 4> {

        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;
        SymTensor3d_4<T> t4;

        static constexpr u32 num_component
            = SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt
              + SymTensor3d_3<T>::compo_cnt + SymTensor3d_4<T>::compo_cnt;

        static constexpr u32 offset_t1 = 0;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;
        static constexpr u32 offset_t4 = offset_t3 + SymTensor3d_3<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            auto A4 = SymTensor3d_4<T>{
                A3.v_000 * A1.v_0,
                A3.v_001 * A1.v_0,
                A3.v_002 * A1.v_0,
                A3.v_011 * A1.v_0,
                A3.v_012 * A1.v_0,
                A3.v_022 * A1.v_0,
                A3.v_111 * A1.v_0,
                A3.v_112 * A1.v_0,
                A3.v_122 * A1.v_0,
                A3.v_222 * A1.v_0,
                A3.v_111 * A1.v_1,
                A3.v_112 * A1.v_1,
                A3.v_122 * A1.v_1,
                A3.v_222 * A1.v_1,
                A3.v_222 * A1.v_2};

            return {A1, A2, A3, A4};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto A4 = SymTensor3d_4<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {A1, A2, A3, A4};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {

            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
            t4.store(acc, offset + offset_t4);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3),
                SymTensor3d_4<T>::load(acc, offset + offset_t4)};
        }

        inline SymTensorCollection<T, 1, 4> &operator*=(const T scal) {

            t1 *= scal;
            t2 *= scal;
            t3 *= scal;
            t4 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 1, 4> &operator+=(const SymTensorCollection<T, 1, 4> other) {

            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;
            t4 += other.t4;

            return *this;
        }

        inline SymTensorCollection<T, 1, 4> operator-(
            const SymTensorCollection<T, 1, 4> &other) const {
            return {t1 - other.t1, t2 - other.t2, t3 - other.t3, t4 - other.t4};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 1, 3> {

        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;
        SymTensor3d_3<T> t3;

        static constexpr u32 num_component = SymTensor3d_1<T>::compo_cnt
                                             + SymTensor3d_2<T>::compo_cnt
                                             + SymTensor3d_3<T>::compo_cnt;

        static constexpr u32 offset_t1 = 0;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;
        static constexpr u32 offset_t3 = offset_t2 + SymTensor3d_2<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            auto A3 = SymTensor3d_3<T>{
                A2.v_00 * A1.v_0,
                A2.v_01 * A1.v_0,
                A2.v_02 * A1.v_0,
                A2.v_11 * A1.v_0,
                A2.v_12 * A1.v_0,
                A2.v_22 * A1.v_0,
                A2.v_11 * A1.v_1,
                A2.v_12 * A1.v_1,
                A2.v_22 * A1.v_1,
                A2.v_22 * A1.v_2};

            return {A1, A2, A3};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            auto A3 = SymTensor3d_3<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            return {A1, A2, A3};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {

            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
            t3.store(acc, offset + offset_t3);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2),
                SymTensor3d_3<T>::load(acc, offset + offset_t3)};
        }

        inline SymTensorCollection<T, 1, 3> &operator*=(const T scal) {

            t1 *= scal;
            t2 *= scal;
            t3 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 1, 3> &operator+=(const SymTensorCollection<T, 1, 3> other) {

            t1 += other.t1;
            t2 += other.t2;
            t3 += other.t3;

            return *this;
        }

        inline SymTensorCollection<T, 1, 3> operator-(
            const SymTensorCollection<T, 1, 3> &other) const {
            return {t1 - other.t1, t2 - other.t2, t3 - other.t3};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 1, 2> {

        SymTensor3d_1<T> t1;
        SymTensor3d_2<T> t2;

        static constexpr u32 num_component
            = SymTensor3d_1<T>::compo_cnt + SymTensor3d_2<T>::compo_cnt;

        static constexpr u32 offset_t1 = 0;
        static constexpr u32 offset_t2 = offset_t1 + SymTensor3d_1<T>::compo_cnt;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            auto A2 = SymTensor3d_2<T>{
                A1.v_0 * A1.v_0,
                A1.v_1 * A1.v_0,
                A1.v_2 * A1.v_0,
                A1.v_1 * A1.v_1,
                A1.v_2 * A1.v_1,
                A1.v_2 * A1.v_2,
            };

            return {A1, A2};
        }

        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            auto A2 = SymTensor3d_2<T>{
                0,
                0,
                0,
                0,
                0,
                0,
            };

            return {A1, A2};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {

            t1.store(acc, offset + offset_t1);
            t2.store(acc, offset + offset_t2);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{
                SymTensor3d_1<T>::load(acc, offset + offset_t1),
                SymTensor3d_2<T>::load(acc, offset + offset_t2)};
        }

        inline SymTensorCollection<T, 1, 2> &operator*=(const T scal) {

            t1 *= scal;
            t2 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 1, 2> &operator+=(const SymTensorCollection<T, 1, 2> other) {

            t1 += other.t1;
            t2 += other.t2;

            return *this;
        }

        inline SymTensorCollection<T, 1, 2> operator-(
            const SymTensorCollection<T, 1, 2> &other) const {
            return {t1 - other.t1, t2 - other.t2};
        }
    };

    template<class T>
    struct SymTensorCollection<T, 1, 1> {

        SymTensor3d_1<T> t1;

        static constexpr u32 num_component = SymTensor3d_1<T>::compo_cnt;

        static constexpr u32 offset_t1 = 0;

        inline static SymTensorCollection from_vec(const sycl::vec<T, 3> &v) {
            auto A1 = SymTensor3d_1<T>{v.x(), v.y(), v.z()};

            return {A1};
        }
        inline static SymTensorCollection zeros() {
            auto A1 = SymTensor3d_1<T>{0, 0, 0};

            return {A1};
        }

        template<class Tacc>
        inline void store(Tacc &&acc, u32 offset) const {

            t1.store(acc, offset + offset_t1);
        }

        template<class Tacc>
        inline static SymTensorCollection load(Tacc &&acc, u32 offset) {
            return SymTensorCollection{SymTensor3d_1<T>::load(acc, offset + offset_t1)};
        }

        inline SymTensorCollection<T, 1, 1> &operator*=(const T scal) {

            t1 *= scal;

            return *this;
        }

        inline SymTensorCollection<T, 1, 1> &operator+=(const SymTensorCollection<T, 1, 1> other) {

            t1 += other.t1;

            return *this;
        }

        inline SymTensorCollection<T, 1, 1> operator-(
            const SymTensorCollection<T, 1, 1> &other) const {
            return {t1 - other.t1};
        }
    };
} // namespace shammath

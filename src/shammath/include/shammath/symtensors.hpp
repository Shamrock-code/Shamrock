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
 * @file symtensors.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"

namespace shammath {

    template<class T>
    struct SymTensor3d_1 {

        static constexpr u32 compo_cnt = 3;

        T v_0;
        T v_1;
        T v_2;

        inline T inner(const SymTensor3d_1 &t) const {
            return v_0 * t.v_0 + v_1 * t.v_1 + v_2 * t.v_2;
        }

        inline SymTensor3d_1 inner(const T scal) const {
            return SymTensor3d_1<T>{v_0 * scal, v_1 * scal, v_2 * scal};
        }

        inline SymTensor3d_1 &operator*=(const T scal) {

            v_0 *= scal;
            v_1 *= scal;
            v_2 *= scal;

            return *this;
        }

        SymTensor3d_1 operator*(const T &scal) const {
            return SymTensor3d_1<T>{
                v_0 * scal,
                v_1 * scal,
                v_2 * scal,
            };
        }

        inline SymTensor3d_1 &operator+=(const SymTensor3d_1 other) {

            v_0 += other.v_0;
            v_1 += other.v_1;
            v_2 += other.v_2;

            return *this;
        }

        SymTensor3d_1 operator+(const SymTensor3d_1 &t2) const {
            return SymTensor3d_1<T>{v_0 + t2.v_0, v_1 + t2.v_1, v_2 + t2.v_2};
        }

        SymTensor3d_1 operator-(const SymTensor3d_1 &t2) const {
            return SymTensor3d_1<T>{v_0 - t2.v_0, v_1 - t2.v_1, v_2 - t2.v_2};
        }

        template<class Tacc>
        inline void store(Tacc &acc, u32 offset) {
            acc[offset + 0] = v_0;
            acc[offset + 1] = v_1;
            acc[offset + 2] = v_2;
        }

        template<class Tacc>
        inline static SymTensor3d_1 load(Tacc &acc, u32 offset) {
            return SymTensor3d_1{
                acc[offset + 0],
                acc[offset + 1],
                acc[offset + 2],
            };
        }
    };

    template<class T>
    struct SymTensor3d_2 {

        static constexpr u32 compo_cnt = 6;

        T v_00;
        T v_01;
        T v_02;
        T v_11;
        T v_12;
        T v_22;

        inline T inner(const SymTensor3d_2 &t) const {
            return v_00 * t.v_00 + 2 * v_01 * t.v_01 + 2 * v_02 * t.v_02 + v_11 * t.v_11
                   + 2 * v_12 * t.v_12 + v_22 * t.v_22;
        }

        inline SymTensor3d_1<T> inner(const SymTensor3d_1<T> &t) const {
            return SymTensor3d_1<T>{
                v_00 * t.v_0 + v_01 * t.v_1 + v_02 * t.v_2,
                v_01 * t.v_0 + v_11 * t.v_1 + v_12 * t.v_2,
                v_02 * t.v_0 + v_12 * t.v_1 + v_22 * t.v_2};
        }

        inline SymTensor3d_2 inner(const T scal) const {
            return SymTensor3d_2<T>{
                v_00 * scal, v_01 * scal, v_02 * scal, v_11 * scal, v_12 * scal, v_22 * scal};
        }

        inline SymTensor3d_2 &operator*=(const T scal) {

            v_00 *= scal;
            v_01 *= scal;
            v_02 *= scal;
            v_11 *= scal;
            v_12 *= scal;
            v_22 *= scal;

            return *this;
        }

        SymTensor3d_2 operator*(const T &scal) const {
            return SymTensor3d_2<T>{
                v_00 * scal, v_01 * scal, v_02 * scal, v_11 * scal, v_12 * scal, v_22 * scal};
        }

        inline SymTensor3d_2 &operator+=(const SymTensor3d_2 other) {

            v_00 += other.v_00;
            v_01 += other.v_01;
            v_02 += other.v_02;
            v_11 += other.v_11;
            v_12 += other.v_12;
            v_22 += other.v_22;

            return *this;
        }

        SymTensor3d_2 operator+(const SymTensor3d_2 &t2) const {
            return SymTensor3d_2<T>{
                v_00 + t2.v_00,
                v_01 + t2.v_01,
                v_02 + t2.v_02,
                v_11 + t2.v_11,
                v_12 + t2.v_12,
                v_22 + t2.v_22};
        }

        SymTensor3d_2 operator-(const SymTensor3d_2 &t2) const {
            return SymTensor3d_2<T>{
                v_00 - t2.v_00,
                v_01 - t2.v_01,
                v_02 - t2.v_02,
                v_11 - t2.v_11,
                v_12 - t2.v_12,
                v_22 - t2.v_22};
        }

        template<class Tacc>
        inline void store(Tacc &acc, u32 offset) {
            acc[offset + 0] = v_00;
            acc[offset + 1] = v_01;
            acc[offset + 2] = v_02;
            acc[offset + 3] = v_11;
            acc[offset + 4] = v_12;
            acc[offset + 5] = v_22;
        }

        template<class Tacc>
        inline static SymTensor3d_2 load(Tacc &acc, u32 offset) {
            return SymTensor3d_2{
                acc[offset + 0],
                acc[offset + 1],
                acc[offset + 2],
                acc[offset + 3],
                acc[offset + 4],
                acc[offset + 5],
            };
        }
    };

    template<class T>
    struct SymTensor3d_3 {

        static constexpr u32 compo_cnt = 10;

        T v_000;
        T v_001;
        T v_002;
        T v_011;
        T v_012;
        T v_022;
        T v_111;
        T v_112;
        T v_122;
        T v_222;

        inline T inner(const SymTensor3d_3 &t) const {
            return v_000 * t.v_000 + 3 * v_001 * t.v_001 + 3 * v_002 * t.v_002 + 3 * v_011 * t.v_011
                   + 6 * v_012 * t.v_012 + 3 * v_022 * t.v_022 + v_111 * t.v_111
                   + 3 * v_112 * t.v_112 + 3 * v_122 * t.v_122 + v_222 * t.v_222;
        }

        inline SymTensor3d_1<T> inner(const SymTensor3d_2<T> &t) const {
            return SymTensor3d_1<T>{
                v_000 * t.v_00 + 2 * v_001 * t.v_01 + 2 * v_002 * t.v_02 + v_011 * t.v_11
                    + 2 * v_012 * t.v_12 + v_022 * t.v_22,
                v_001 * t.v_00 + 2 * v_011 * t.v_01 + 2 * v_012 * t.v_02 + v_111 * t.v_11
                    + 2 * v_112 * t.v_12 + v_122 * t.v_22,
                v_002 * t.v_00 + 2 * v_012 * t.v_01 + 2 * v_022 * t.v_02 + v_112 * t.v_11
                    + 2 * v_122 * t.v_12 + v_222 * t.v_22};
        }

        inline SymTensor3d_2<T> inner(const SymTensor3d_1<T> &t) const {

            return SymTensor3d_2<T>{
                v_000 * t.v_0 + v_001 * t.v_1 + v_002 * t.v_2,
                v_001 * t.v_0 + v_011 * t.v_1 + v_012 * t.v_2,
                v_002 * t.v_0 + v_012 * t.v_1 + v_022 * t.v_2,
                v_011 * t.v_0 + v_111 * t.v_1 + v_112 * t.v_2,
                v_012 * t.v_0 + v_112 * t.v_1 + v_122 * t.v_2,
                v_022 * t.v_0 + v_122 * t.v_1 + v_222 * t.v_2};
        }

        inline SymTensor3d_3 inner(const T scal) const {
            return SymTensor3d_3<T>{
                v_000 * scal,
                v_001 * scal,
                v_002 * scal,
                v_011 * scal,
                v_012 * scal,
                v_022 * scal,
                v_111 * scal,
                v_112 * scal,
                v_122 * scal,
                v_222 * scal};
        }

        inline SymTensor3d_3 &operator*=(const T scal) {

            v_000 *= scal;
            v_001 *= scal;
            v_002 *= scal;
            v_011 *= scal;
            v_012 *= scal;
            v_022 *= scal;
            v_111 *= scal;
            v_112 *= scal;
            v_122 *= scal;
            v_222 *= scal;

            return *this;
        }

        SymTensor3d_3 operator*(const T &scal) const {
            return SymTensor3d_3{
                v_000 * scal,
                v_001 * scal,
                v_002 * scal,
                v_011 * scal,
                v_012 * scal,
                v_022 * scal,
                v_111 * scal,
                v_112 * scal,
                v_122 * scal,
                v_222 * scal};
        }

        inline SymTensor3d_3 &operator+=(const SymTensor3d_3 other) {

            v_000 += other.v_000;
            v_001 += other.v_001;
            v_002 += other.v_002;
            v_011 += other.v_011;
            v_012 += other.v_012;
            v_022 += other.v_022;
            v_111 += other.v_111;
            v_112 += other.v_112;
            v_122 += other.v_122;
            v_222 += other.v_222;

            return *this;
        }

        SymTensor3d_3 operator+(const SymTensor3d_3 &t2) const {
            return SymTensor3d_3{
                v_000 + t2.v_000,
                v_001 + t2.v_001,
                v_002 + t2.v_002,
                v_011 + t2.v_011,
                v_012 + t2.v_012,
                v_022 + t2.v_022,
                v_111 + t2.v_111,
                v_112 + t2.v_112,
                v_122 + t2.v_122,
                v_222 + t2.v_222};
        }

        SymTensor3d_3 operator-(const SymTensor3d_3 &t2) const {
            return SymTensor3d_3{
                v_000 - t2.v_000,
                v_001 - t2.v_001,
                v_002 - t2.v_002,
                v_011 - t2.v_011,
                v_012 - t2.v_012,
                v_022 - t2.v_022,
                v_111 - t2.v_111,
                v_112 - t2.v_112,
                v_122 - t2.v_122,
                v_222 - t2.v_222};
        }

        template<class Tacc>
        inline void store(Tacc &acc, u32 offset) {
            acc[offset + 0] = v_000;
            acc[offset + 1] = v_001;
            acc[offset + 2] = v_002;
            acc[offset + 3] = v_011;
            acc[offset + 4] = v_012;
            acc[offset + 5] = v_022;
            acc[offset + 6] = v_111;
            acc[offset + 7] = v_112;
            acc[offset + 8] = v_122;
            acc[offset + 9] = v_222;
        }

        template<class Tacc>
        inline static SymTensor3d_3 load(Tacc &acc, u32 offset) {
            return SymTensor3d_3{
                acc[offset + 0],
                acc[offset + 1],
                acc[offset + 2],
                acc[offset + 3],
                acc[offset + 4],
                acc[offset + 5],
                acc[offset + 6],
                acc[offset + 7],
                acc[offset + 8],
                acc[offset + 9],
            };
        }
    };

    template<class T>
    struct SymTensor3d_4 {

        static constexpr u32 compo_cnt = 15;

        T v_0000;
        T v_0001;
        T v_0002;
        T v_0011;
        T v_0012;
        T v_0022;
        T v_0111;
        T v_0112;
        T v_0122;
        T v_0222;
        T v_1111;
        T v_1112;
        T v_1122;
        T v_1222;
        T v_2222;

        inline T inner(const SymTensor3d_4 &t) const {
            return v_0000 * t.v_0000 + 4 * v_0001 * t.v_0001 + 4 * v_0002 * t.v_0002
                   + 6 * v_0011 * t.v_0011 + 12 * v_0012 * t.v_0012 + 6 * v_0022 * t.v_0022
                   + 4 * v_0111 * t.v_0111 + 12 * v_0112 * t.v_0112 + 12 * v_0122 * t.v_0122
                   + 4 * v_0222 * t.v_0222 + v_1111 * t.v_1111 + 4 * v_1112 * t.v_1112
                   + 6 * v_1122 * t.v_1122 + 4 * v_1222 * t.v_1222 + v_2222 * t.v_2222;
        }

        inline SymTensor3d_1<T> inner(const SymTensor3d_3<T> &t) const {
            return SymTensor3d_1<T>{
                v_0000 * t.v_000 + 3 * v_0001 * t.v_001 + 3 * v_0002 * t.v_002
                    + 3 * v_0011 * t.v_011 + 6 * v_0012 * t.v_012 + 3 * v_0022 * t.v_022
                    + v_0111 * t.v_111 + 3 * v_0112 * t.v_112 + 3 * v_0122 * t.v_122
                    + v_0222 * t.v_222,
                v_0001 * t.v_000 + 3 * v_0011 * t.v_001 + 3 * v_0012 * t.v_002
                    + 3 * v_0111 * t.v_011 + 6 * v_0112 * t.v_012 + 3 * v_0122 * t.v_022
                    + v_1111 * t.v_111 + 3 * v_1112 * t.v_112 + 3 * v_1122 * t.v_122
                    + v_1222 * t.v_222,
                v_0002 * t.v_000 + 3 * v_0012 * t.v_001 + 3 * v_0022 * t.v_002
                    + 3 * v_0112 * t.v_011 + 6 * v_0122 * t.v_012 + 3 * v_0222 * t.v_022
                    + v_1112 * t.v_111 + 3 * v_1122 * t.v_112 + 3 * v_1222 * t.v_122
                    + v_2222 * t.v_222};
        }

        inline SymTensor3d_2<T> inner(const SymTensor3d_2<T> &t) const {

            return SymTensor3d_2<T>{
                v_0000 * t.v_00 + 2 * v_0001 * t.v_01 + 2 * v_0002 * t.v_02 + v_0011 * t.v_11
                    + 2 * v_0012 * t.v_12 + v_0022 * t.v_22,
                v_0001 * t.v_00 + 2 * v_0011 * t.v_01 + 2 * v_0012 * t.v_02 + v_0111 * t.v_11
                    + 2 * v_0112 * t.v_12 + v_0122 * t.v_22,
                v_0002 * t.v_00 + 2 * v_0012 * t.v_01 + 2 * v_0022 * t.v_02 + v_0112 * t.v_11
                    + 2 * v_0122 * t.v_12 + v_0222 * t.v_22,
                v_0011 * t.v_00 + 2 * v_0111 * t.v_01 + 2 * v_0112 * t.v_02 + v_1111 * t.v_11
                    + 2 * v_1112 * t.v_12 + v_1122 * t.v_22,
                v_0012 * t.v_00 + 2 * v_0112 * t.v_01 + 2 * v_0122 * t.v_02 + v_1112 * t.v_11
                    + 2 * v_1122 * t.v_12 + v_1222 * t.v_22,
                v_0022 * t.v_00 + 2 * v_0122 * t.v_01 + 2 * v_0222 * t.v_02 + v_1122 * t.v_11
                    + 2 * v_1222 * t.v_12 + v_2222 * t.v_22};
        }

        inline SymTensor3d_3<T> inner(const SymTensor3d_1<T> &t) const {
            return SymTensor3d_3<T>{
                v_0000 * t.v_0 + v_0001 * t.v_1 + v_0002 * t.v_2,
                v_0001 * t.v_0 + v_0011 * t.v_1 + v_0012 * t.v_2,
                v_0002 * t.v_0 + v_0012 * t.v_1 + v_0022 * t.v_2,
                v_0011 * t.v_0 + v_0111 * t.v_1 + v_0112 * t.v_2,
                v_0012 * t.v_0 + v_0112 * t.v_1 + v_0122 * t.v_2,
                v_0022 * t.v_0 + v_0122 * t.v_1 + v_0222 * t.v_2,
                v_0111 * t.v_0 + v_1111 * t.v_1 + v_1112 * t.v_2,
                v_0112 * t.v_0 + v_1112 * t.v_1 + v_1122 * t.v_2,
                v_0122 * t.v_0 + v_1122 * t.v_1 + v_1222 * t.v_2,
                v_0222 * t.v_0 + v_1222 * t.v_1 + v_2222 * t.v_2};
        }

        inline SymTensor3d_4 inner(const T scal) const {
            return SymTensor3d_4<T>{
                v_0000 * scal,
                v_0001 * scal,
                v_0002 * scal,
                v_0011 * scal,
                v_0012 * scal,
                v_0022 * scal,
                v_0111 * scal,
                v_0112 * scal,
                v_0122 * scal,
                v_0222 * scal,
                v_1111 * scal,
                v_1112 * scal,
                v_1122 * scal,
                v_1222 * scal,
                v_2222 * scal};
        }

        inline SymTensor3d_4 &operator*=(const T scal) {

            v_0000 *= scal;
            v_0001 *= scal;
            v_0002 *= scal;
            v_0011 *= scal;
            v_0012 *= scal;
            v_0022 *= scal;
            v_0111 *= scal;
            v_0112 *= scal;
            v_0122 *= scal;
            v_0222 *= scal;
            v_1111 *= scal;
            v_1112 *= scal;
            v_1122 *= scal;
            v_1222 *= scal;
            v_2222 *= scal;

            return *this;
        }

        SymTensor3d_4 operator*(const T &scal) const {
            return SymTensor3d_4{
                v_0000 * scal,
                v_0001 * scal,
                v_0002 * scal,
                v_0011 * scal,
                v_0012 * scal,
                v_0022 * scal,
                v_0111 * scal,
                v_0112 * scal,
                v_0122 * scal,
                v_0222 * scal,
                v_1111 * scal,
                v_1112 * scal,
                v_1122 * scal,
                v_1222 * scal,
                v_2222 * scal};
        }

        inline SymTensor3d_4 &operator+=(const SymTensor3d_4 other) {

            v_0000 += other.v_0000;
            v_0001 += other.v_0001;
            v_0002 += other.v_0002;
            v_0011 += other.v_0011;
            v_0012 += other.v_0012;
            v_0022 += other.v_0022;
            v_0111 += other.v_0111;
            v_0112 += other.v_0112;
            v_0122 += other.v_0122;
            v_0222 += other.v_0222;
            v_1111 += other.v_1111;
            v_1112 += other.v_1112;
            v_1122 += other.v_1122;
            v_1222 += other.v_1222;
            v_2222 += other.v_2222;

            return *this;
        }

        SymTensor3d_4 operator+(const SymTensor3d_4 &t2) const {
            return SymTensor3d_4{
                v_0000 + t2.v_0000,
                v_0001 + t2.v_0001,
                v_0002 + t2.v_0002,
                v_0011 + t2.v_0011,
                v_0012 + t2.v_0012,
                v_0022 + t2.v_0022,
                v_0111 + t2.v_0111,
                v_0112 + t2.v_0112,
                v_0122 + t2.v_0122,
                v_0222 + t2.v_0222,
                v_1111 + t2.v_1111,
                v_1112 + t2.v_1112,
                v_1122 + t2.v_1122,
                v_1222 + t2.v_1222,
                v_2222 + t2.v_2222};
        }

        SymTensor3d_4 operator-(const SymTensor3d_4 &t2) const {
            return SymTensor3d_4{
                v_0000 - t2.v_0000,
                v_0001 - t2.v_0001,
                v_0002 - t2.v_0002,
                v_0011 - t2.v_0011,
                v_0012 - t2.v_0012,
                v_0022 - t2.v_0022,
                v_0111 - t2.v_0111,
                v_0112 - t2.v_0112,
                v_0122 - t2.v_0122,
                v_0222 - t2.v_0222,
                v_1111 - t2.v_1111,
                v_1112 - t2.v_1112,
                v_1122 - t2.v_1122,
                v_1222 - t2.v_1222,
                v_2222 - t2.v_2222};
        }

        template<class Tacc>
        inline void store(Tacc &acc, u32 offset) {
            acc[offset + 0]  = v_0000;
            acc[offset + 1]  = v_0001;
            acc[offset + 2]  = v_0002;
            acc[offset + 3]  = v_0011;
            acc[offset + 4]  = v_0012;
            acc[offset + 5]  = v_0022;
            acc[offset + 6]  = v_0111;
            acc[offset + 7]  = v_0112;
            acc[offset + 8]  = v_0122;
            acc[offset + 9]  = v_0222;
            acc[offset + 10] = v_1111;
            acc[offset + 11] = v_1112;
            acc[offset + 12] = v_1122;
            acc[offset + 13] = v_1222;
            acc[offset + 14] = v_2222;
        }

        template<class Tacc>
        inline static SymTensor3d_4 load(Tacc &acc, u32 offset) {
            return SymTensor3d_4<T>{
                acc[offset + 0],
                acc[offset + 1],
                acc[offset + 2],
                acc[offset + 3],
                acc[offset + 4],
                acc[offset + 5],
                acc[offset + 6],
                acc[offset + 7],
                acc[offset + 8],
                acc[offset + 9],
                acc[offset + 10],
                acc[offset + 11],
                acc[offset + 12],
                acc[offset + 13],
                acc[offset + 14],
            };
        }
    };

    template<class T>
    struct SymTensor3d_5 {

        static constexpr u32 compo_cnt = 21;

        T v_00000;
        T v_00001;
        T v_00002;
        T v_00011;
        T v_00012;
        T v_00022;
        T v_00111;
        T v_00112;
        T v_00122;
        T v_00222;
        T v_01111;
        T v_01112;
        T v_01122;
        T v_01222;
        T v_02222;
        T v_11111;
        T v_11112;
        T v_11122;
        T v_11222;
        T v_12222;
        T v_22222;

        inline T inner(const SymTensor3d_5 &t) const {
            return v_00000 * t.v_00000 + 5 * v_00001 * t.v_00001 + 5 * v_00002 * t.v_00002
                   + 10 * v_00011 * t.v_00011 + 20 * v_00012 * t.v_00012 + 10 * v_00022 * t.v_00022
                   + 10 * v_00111 * t.v_00111 + 30 * v_00112 * t.v_00112 + 30 * v_00122 * t.v_00122
                   + 10 * v_00222 * t.v_00222 + 5 * v_01111 * t.v_01111 + 20 * v_01112 * t.v_01112
                   + 30 * v_01122 * t.v_01122 + 20 * v_01222 * t.v_01222 + 5 * v_02222 * t.v_02222
                   + v_11111 * t.v_11111
                   + 5
                         * (v_11112 * t.v_11112 + 2 * v_11122 * t.v_11122 + 2 * v_11222 * t.v_11222
                            + v_12222 * t.v_12222)
                   + v_22222 * t.v_22222;
        }

        inline SymTensor3d_1<T> inner(const SymTensor3d_4<T> &t) const {
            return SymTensor3d_1<T>{
                v_00000 * t.v_0000 + 4 * v_00001 * t.v_0001 + 4 * v_00002 * t.v_0002
                    + 6 * v_00011 * t.v_0011 + 12 * v_00012 * t.v_0012 + 6 * v_00022 * t.v_0022
                    + 4 * v_00111 * t.v_0111 + 12 * v_00112 * t.v_0112 + 12 * v_00122 * t.v_0122
                    + 4 * v_00222 * t.v_0222 + v_01111 * t.v_1111 + 4 * v_01112 * t.v_1112
                    + 6 * v_01122 * t.v_1122 + 4 * v_01222 * t.v_1222 + v_02222 * t.v_2222,
                v_00001 * t.v_0000 + 4 * v_00011 * t.v_0001 + 4 * v_00012 * t.v_0002
                    + 6 * v_00111 * t.v_0011 + 12 * v_00112 * t.v_0012 + 6 * v_00122 * t.v_0022
                    + 4 * v_01111 * t.v_0111 + 12 * v_01112 * t.v_0112 + 12 * v_01122 * t.v_0122
                    + 4 * v_01222 * t.v_0222 + v_11111 * t.v_1111 + 4 * v_11112 * t.v_1112
                    + 6 * v_11122 * t.v_1122 + 4 * v_11222 * t.v_1222 + v_12222 * t.v_2222,
                v_00002 * t.v_0000 + 4 * v_00012 * t.v_0001 + 4 * v_00022 * t.v_0002
                    + 6 * v_00112 * t.v_0011 + 12 * v_00122 * t.v_0012 + 6 * v_00222 * t.v_0022
                    + 4 * v_01112 * t.v_0111 + 12 * v_01122 * t.v_0112 + 12 * v_01222 * t.v_0122
                    + 4 * v_02222 * t.v_0222 + v_11112 * t.v_1111 + 4 * v_11122 * t.v_1112
                    + 6 * v_11222 * t.v_1122 + 4 * v_12222 * t.v_1222 + v_22222 * t.v_2222};
        }

        inline SymTensor3d_2<T> inner(const SymTensor3d_3<T> &t) const {
            return SymTensor3d_2<T>{
                v_00000 * t.v_000 + 3 * v_00001 * t.v_001 + 3 * v_00002 * t.v_002
                    + 3 * v_00011 * t.v_011 + 6 * v_00012 * t.v_012 + 3 * v_00022 * t.v_022
                    + v_00111 * t.v_111 + 3 * v_00112 * t.v_112 + 3 * v_00122 * t.v_122
                    + v_00222 * t.v_222,
                v_00001 * t.v_000 + 3 * v_00011 * t.v_001 + 3 * v_00012 * t.v_002
                    + 3 * v_00111 * t.v_011 + 6 * v_00112 * t.v_012 + 3 * v_00122 * t.v_022
                    + v_01111 * t.v_111 + 3 * v_01112 * t.v_112 + 3 * v_01122 * t.v_122
                    + v_01222 * t.v_222,
                v_00002 * t.v_000 + 3 * v_00012 * t.v_001 + 3 * v_00022 * t.v_002
                    + 3 * v_00112 * t.v_011 + 6 * v_00122 * t.v_012 + 3 * v_00222 * t.v_022
                    + v_01112 * t.v_111 + 3 * v_01122 * t.v_112 + 3 * v_01222 * t.v_122
                    + v_02222 * t.v_222,
                v_00011 * t.v_000 + 3 * v_00111 * t.v_001 + 3 * v_00112 * t.v_002
                    + 3 * v_01111 * t.v_011 + 6 * v_01112 * t.v_012 + 3 * v_01122 * t.v_022
                    + v_11111 * t.v_111 + 3 * v_11112 * t.v_112 + 3 * v_11122 * t.v_122
                    + v_11222 * t.v_222,
                v_00012 * t.v_000 + 3 * v_00112 * t.v_001 + 3 * v_00122 * t.v_002
                    + 3 * v_01112 * t.v_011 + 6 * v_01122 * t.v_012 + 3 * v_01222 * t.v_022
                    + v_11112 * t.v_111 + 3 * v_11122 * t.v_112 + 3 * v_11222 * t.v_122
                    + v_12222 * t.v_222,
                v_00022 * t.v_000 + 3 * v_00122 * t.v_001 + 3 * v_00222 * t.v_002
                    + 3 * v_01122 * t.v_011 + 6 * v_01222 * t.v_012 + 3 * v_02222 * t.v_022
                    + v_11122 * t.v_111 + 3 * v_11222 * t.v_112 + 3 * v_12222 * t.v_122
                    + v_22222 * t.v_222};
        }

        inline SymTensor3d_3<T> inner(const SymTensor3d_2<T> &t) const {
            return SymTensor3d_3<T>{
                v_00000 * t.v_00 + 2 * v_00001 * t.v_01 + 2 * v_00002 * t.v_02 + v_00011 * t.v_11
                    + 2 * v_00012 * t.v_12 + v_00022 * t.v_22,
                v_00001 * t.v_00 + 2 * v_00011 * t.v_01 + 2 * v_00012 * t.v_02 + v_00111 * t.v_11
                    + 2 * v_00112 * t.v_12 + v_00122 * t.v_22,
                v_00002 * t.v_00 + 2 * v_00012 * t.v_01 + 2 * v_00022 * t.v_02 + v_00112 * t.v_11
                    + 2 * v_00122 * t.v_12 + v_00222 * t.v_22,
                v_00011 * t.v_00 + 2 * v_00111 * t.v_01 + 2 * v_00112 * t.v_02 + v_01111 * t.v_11
                    + 2 * v_01112 * t.v_12 + v_01122 * t.v_22,
                v_00012 * t.v_00 + 2 * v_00112 * t.v_01 + 2 * v_00122 * t.v_02 + v_01112 * t.v_11
                    + 2 * v_01122 * t.v_12 + v_01222 * t.v_22,
                v_00022 * t.v_00 + 2 * v_00122 * t.v_01 + 2 * v_00222 * t.v_02 + v_01122 * t.v_11
                    + 2 * v_01222 * t.v_12 + v_02222 * t.v_22,
                v_00111 * t.v_00 + 2 * v_01111 * t.v_01 + 2 * v_01112 * t.v_02 + v_11111 * t.v_11
                    + 2 * v_11112 * t.v_12 + v_11122 * t.v_22,
                v_00112 * t.v_00 + 2 * v_01112 * t.v_01 + 2 * v_01122 * t.v_02 + v_11112 * t.v_11
                    + 2 * v_11122 * t.v_12 + v_11222 * t.v_22,
                v_00122 * t.v_00 + 2 * v_01122 * t.v_01 + 2 * v_01222 * t.v_02 + v_11122 * t.v_11
                    + 2 * v_11222 * t.v_12 + v_12222 * t.v_22,
                v_00222 * t.v_00 + 2 * v_01222 * t.v_01 + 2 * v_02222 * t.v_02 + v_11222 * t.v_11
                    + 2 * v_12222 * t.v_12 + v_22222 * t.v_22};
        }

        inline SymTensor3d_4<T> inner(const SymTensor3d_1<T> &t) const {

            return SymTensor3d_4<T>{
                v_00000 * t.v_0 + v_00001 * t.v_1 + v_00002 * t.v_2,
                v_00001 * t.v_0 + v_00011 * t.v_1 + v_00012 * t.v_2,
                v_00002 * t.v_0 + v_00012 * t.v_1 + v_00022 * t.v_2,
                v_00011 * t.v_0 + v_00111 * t.v_1 + v_00112 * t.v_2,
                v_00012 * t.v_0 + v_00112 * t.v_1 + v_00122 * t.v_2,
                v_00022 * t.v_0 + v_00122 * t.v_1 + v_00222 * t.v_2,
                v_00111 * t.v_0 + v_01111 * t.v_1 + v_01112 * t.v_2,
                v_00112 * t.v_0 + v_01112 * t.v_1 + v_01122 * t.v_2,
                v_00122 * t.v_0 + v_01122 * t.v_1 + v_01222 * t.v_2,
                v_00222 * t.v_0 + v_01222 * t.v_1 + v_02222 * t.v_2,
                v_01111 * t.v_0 + v_11111 * t.v_1 + v_11112 * t.v_2,
                v_01112 * t.v_0 + v_11112 * t.v_1 + v_11122 * t.v_2,
                v_01122 * t.v_0 + v_11122 * t.v_1 + v_11222 * t.v_2,
                v_01222 * t.v_0 + v_11222 * t.v_1 + v_12222 * t.v_2,
                v_02222 * t.v_0 + v_12222 * t.v_1 + v_22222 * t.v_2};
        }

        inline SymTensor3d_5 inner(const T scal) const {
            return SymTensor3d_5<T>{v_00000 * scal, v_00001 * scal, v_00002 * scal, v_00011 * scal,
                                    v_00012 * scal, v_00022 * scal, v_00111 * scal, v_00112 * scal,
                                    v_00122 * scal, v_00222 * scal, v_01111 * scal, v_01112 * scal,
                                    v_01122 * scal, v_01222 * scal, v_02222 * scal, v_11111 * scal,
                                    v_11112 * scal, v_11122 * scal, v_11222 * scal, v_12222 * scal,
                                    v_22222 * scal};
        }

        inline SymTensor3d_5 &operator*=(const T scal) {

            v_00000 *= scal;
            v_00001 *= scal;
            v_00002 *= scal;
            v_00011 *= scal;
            v_00012 *= scal;
            v_00022 *= scal;
            v_00111 *= scal;
            v_00112 *= scal;
            v_00122 *= scal;
            v_00222 *= scal;
            v_01111 *= scal;
            v_01112 *= scal;
            v_01122 *= scal;
            v_01222 *= scal;
            v_02222 *= scal;
            v_11111 *= scal;
            v_11112 *= scal;
            v_11122 *= scal;
            v_11222 *= scal;
            v_12222 *= scal;
            v_22222 *= scal;

            return *this;
        }

        SymTensor3d_5 operator*(const T &scal) const {
            return SymTensor3d_5<T>{v_00000 * scal, v_00001 * scal, v_00002 * scal, v_00011 * scal,
                                    v_00012 * scal, v_00022 * scal, v_00111 * scal, v_00112 * scal,
                                    v_00122 * scal, v_00222 * scal, v_01111 * scal, v_01112 * scal,
                                    v_01122 * scal, v_01222 * scal, v_02222 * scal, v_11111 * scal,
                                    v_11112 * scal, v_11122 * scal, v_11222 * scal, v_12222 * scal,
                                    v_22222 * scal};
        }

        inline SymTensor3d_5 &operator+=(const SymTensor3d_5 other) {

            v_00000 += other.v_00000;
            v_00001 += other.v_00001;
            v_00002 += other.v_00002;
            v_00011 += other.v_00011;
            v_00012 += other.v_00012;
            v_00022 += other.v_00022;
            v_00111 += other.v_00111;
            v_00112 += other.v_00112;
            v_00122 += other.v_00122;
            v_00222 += other.v_00222;
            v_01111 += other.v_01111;
            v_01112 += other.v_01112;
            v_01122 += other.v_01122;
            v_01222 += other.v_01222;
            v_02222 += other.v_02222;
            v_11111 += other.v_11111;
            v_11112 += other.v_11112;
            v_11122 += other.v_11122;
            v_11222 += other.v_11222;
            v_12222 += other.v_12222;
            v_22222 += other.v_22222;

            return *this;
        }

        SymTensor3d_5 operator+(const SymTensor3d_5 &t2) const {
            return SymTensor3d_5<T>{
                v_00000 + t2.v_00000, v_00001 + t2.v_00001, v_00002 + t2.v_00002,
                v_00011 + t2.v_00011, v_00012 + t2.v_00012, v_00022 + t2.v_00022,
                v_00111 + t2.v_00111, v_00112 + t2.v_00112, v_00122 + t2.v_00122,
                v_00222 + t2.v_00222, v_01111 + t2.v_01111, v_01112 + t2.v_01112,
                v_01122 + t2.v_01122, v_01222 + t2.v_01222, v_02222 + t2.v_02222,
                v_11111 + t2.v_11111, v_11112 + t2.v_11112, v_11122 + t2.v_11122,
                v_11222 + t2.v_11222, v_12222 + t2.v_12222, v_22222 + t2.v_22222};
        }

        SymTensor3d_5 operator-(const SymTensor3d_5 &t2) const {
            return SymTensor3d_5<T>{
                v_00000 - t2.v_00000, v_00001 - t2.v_00001, v_00002 - t2.v_00002,
                v_00011 - t2.v_00011, v_00012 - t2.v_00012, v_00022 - t2.v_00022,
                v_00111 - t2.v_00111, v_00112 - t2.v_00112, v_00122 - t2.v_00122,
                v_00222 - t2.v_00222, v_01111 - t2.v_01111, v_01112 - t2.v_01112,
                v_01122 - t2.v_01122, v_01222 - t2.v_01222, v_02222 - t2.v_02222,
                v_11111 - t2.v_11111, v_11112 - t2.v_11112, v_11122 - t2.v_11122,
                v_11222 - t2.v_11222, v_12222 - t2.v_12222, v_22222 - t2.v_22222};
        }

        template<class Tacc>
        inline void store(Tacc &acc, u32 offset) {
            acc[offset + 0]  = v_00000;
            acc[offset + 1]  = v_00001;
            acc[offset + 2]  = v_00002;
            acc[offset + 3]  = v_00011;
            acc[offset + 4]  = v_00012;
            acc[offset + 5]  = v_00022;
            acc[offset + 6]  = v_00111;
            acc[offset + 7]  = v_00112;
            acc[offset + 8]  = v_00122;
            acc[offset + 9]  = v_00222;
            acc[offset + 10] = v_01111;
            acc[offset + 11] = v_01112;
            acc[offset + 12] = v_01122;
            acc[offset + 13] = v_01222;
            acc[offset + 14] = v_02222;
            acc[offset + 15] = v_11111;
            acc[offset + 16] = v_11112;
            acc[offset + 17] = v_11122;
            acc[offset + 18] = v_11222;
            acc[offset + 19] = v_12222;
            acc[offset + 20] = v_22222;
        }

        template<class Tacc>
        inline static SymTensor3d_5 load(Tacc &acc, u32 offset) {
            return SymTensor3d_5<T>{acc[offset + 0],  acc[offset + 1],  acc[offset + 2],
                                    acc[offset + 3],  acc[offset + 4],  acc[offset + 5],
                                    acc[offset + 6],  acc[offset + 7],  acc[offset + 8],
                                    acc[offset + 9],  acc[offset + 10], acc[offset + 11],
                                    acc[offset + 12], acc[offset + 13], acc[offset + 14],
                                    acc[offset + 15], acc[offset + 16], acc[offset + 17],
                                    acc[offset + 18], acc[offset + 19], acc[offset + 20]};
        }
    };

    // rank 5 ops
    template<class T>
    T operator*(const SymTensor3d_5<T> &a, const SymTensor3d_5<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_5<T> &a, const SymTensor3d_4<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_2<T> operator*(const SymTensor3d_5<T> &a, const SymTensor3d_3<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_3<T> operator*(const SymTensor3d_5<T> &a, const SymTensor3d_2<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_4<T> operator*(const SymTensor3d_5<T> &a, const SymTensor3d_1<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_5<T> operator*(const T &a, const SymTensor3d_5<T> &b) {
        return b * a;
    }

    // rank 4 ops
    template<class T>
    T operator*(const SymTensor3d_4<T> &a, const SymTensor3d_4<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_4<T> &a, const SymTensor3d_3<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_2<T> operator*(const SymTensor3d_4<T> &a, const SymTensor3d_2<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_3<T> operator*(const SymTensor3d_4<T> &a, const SymTensor3d_1<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_4<T> operator*(const T &a, const SymTensor3d_4<T> &b) {
        return b * a;
    }

    // rank 3 ops
    template<class T>
    T operator*(const SymTensor3d_3<T> &a, const SymTensor3d_3<T> &b) {
        return a.inner(b);
    }
    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_3<T> &a, const SymTensor3d_2<T> &b) {
        return a.inner(b);
    }
    template<class T>
    SymTensor3d_2<T> operator*(const SymTensor3d_3<T> &a, const SymTensor3d_1<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_2<T> &a, const SymTensor3d_3<T> &b) {
        return b * a;
    }
    template<class T>
    SymTensor3d_2<T> operator*(const SymTensor3d_1<T> &a, const SymTensor3d_3<T> &b) {
        return b * a;
    }
    template<class T>
    SymTensor3d_3<T> operator*(const T &a, const SymTensor3d_3<T> &b) {
        return b * a;
    }

    // rank 2

    template<class T>
    T operator*(const SymTensor3d_2<T> &a, const SymTensor3d_2<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_2<T> &a, const SymTensor3d_1<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const SymTensor3d_1<T> &a, const SymTensor3d_2<T> &b) {
        return b.inner(a);
    }

    template<class T>
    SymTensor3d_2<T> operator*(const T &a, const SymTensor3d_2<T> &b) {
        return b * a;
    }

    // rank 1

    template<class T>
    T operator*(const SymTensor3d_1<T> &a, const SymTensor3d_1<T> &b) {
        return a.inner(b);
    }

    template<class T>
    SymTensor3d_1<T> operator*(const T &a, const SymTensor3d_1<T> &b) {
        return b * a;
    }

} // namespace shammath

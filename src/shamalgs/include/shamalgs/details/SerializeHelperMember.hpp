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
 * @file SerializeHelperMember.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <type_traits>
#include <memory>

namespace shamalgs::details {

    template<class T>
    class SerializeHelperMember {
        public:
        static constexpr u64 szrepr = sizeof(T);
        inline static void store(u8 *ptr_store, T val) {
            T *ptr = (T *) ptr_store;
            *ptr   = val;
        }
        inline static T load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return *ptr;
        }
    };

    template<class T>
    class SerializeHelperMember<sycl::vec<T, 2>> {
        public:
        static constexpr i32 n      = 2;
        static constexpr u64 szrepr = sizeof(T) * n;

        inline static void store(u8 *ptr_store, sycl::vec<T, n> val) {
            T *ptr = (T *) ptr_store;
            ptr[0] = val.x();
            ptr[1] = val.y();
        }
        inline static sycl::vec<T, n> load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return sycl::vec<T, n>{ptr[0], ptr[1]};
        }
    };

    template<class T>
    class SerializeHelperMember<sycl::vec<T, 3>> {
        public:
        static constexpr i32 n      = 3;
        static constexpr u64 szrepr = sizeof(T) * n;

        inline static void store(u8 *ptr_store, sycl::vec<T, n> val) {
            T *ptr = (T *) ptr_store;
            ptr[0] = val.x();
            ptr[1] = val.y();
            ptr[2] = val.z();
        }
        inline static sycl::vec<T, n> load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return sycl::vec<T, n>{ptr[0], ptr[1], ptr[2]};
        }
    };

    template<class T>
    class SerializeHelperMember<sycl::vec<T, 4>> {
        public:
        static constexpr i32 n      = 4;
        static constexpr u64 szrepr = sizeof(T) * n;

        inline static void store(u8 *ptr_store, sycl::vec<T, n> val) {
            T *ptr = (T *) ptr_store;
            ptr[0] = val.x();
            ptr[1] = val.y();
            ptr[2] = val.z();
            ptr[3] = val.w();
        }
        inline static sycl::vec<T, n> load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return sycl::vec<T, n>{ptr[0], ptr[1], ptr[2], ptr[3]};
        }
    };

    template<class T>
    class SerializeHelperMember<sycl::vec<T, 8>> {
        public:
        static constexpr i32 n      = 8;
        static constexpr u64 szrepr = sizeof(T) * n;

        inline static void store(u8 *ptr_store, sycl::vec<T, n> val) {
            T *ptr = (T *) ptr_store;
            ptr[0] = val.s0();
            ptr[1] = val.s1();
            ptr[2] = val.s2();
            ptr[3] = val.s3();
            ptr[4] = val.s4();
            ptr[5] = val.s5();
            ptr[6] = val.s6();
            ptr[7] = val.s7();
        }
        inline static sycl::vec<T, n> load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return sycl::vec<T, n>{ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]};
        }
    };

    template<class T>
    class SerializeHelperMember<sycl::vec<T, 16>> {
        public:
        static constexpr i32 n      = 16;
        static constexpr u64 szrepr = sizeof(T) * n;

        inline static void store(u8 *ptr_store, sycl::vec<T, n> val) {
            T *ptr  = (T *) ptr_store;
            ptr[0]  = val.s0();
            ptr[1]  = val.s1();
            ptr[2]  = val.s2();
            ptr[3]  = val.s3();
            ptr[4]  = val.s4();
            ptr[5]  = val.s5();
            ptr[6]  = val.s6();
            ptr[7]  = val.s7();
            ptr[8]  = val.s8();
            ptr[9]  = val.s9();
            ptr[10] = val.sA();
            ptr[11] = val.sB();
            ptr[12] = val.sC();
            ptr[13] = val.sD();
            ptr[14] = val.sE();
            ptr[15] = val.sF();
        }
        inline static sycl::vec<T, n> load(const u8 *ptr_load) {
            T *ptr = (T *) ptr_load;
            return sycl::vec<T, n>{
                ptr[0],
                ptr[1],
                ptr[2],
                ptr[3],
                ptr[4],
                ptr[5],
                ptr[6],
                ptr[7],
                ptr[8],
                ptr[9],
                ptr[10],
                ptr[11],
                ptr[12],
                ptr[13],
                ptr[14],
                ptr[15]};
        }
    };

} // namespace shamalgs::details

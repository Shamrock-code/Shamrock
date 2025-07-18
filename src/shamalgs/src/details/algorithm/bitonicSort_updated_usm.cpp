// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file bitonicSort_updated_usm.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shamalgs/details/algorithm/bitonicSort_updated_usm.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl_utils.hpp"

// modified from http://www.bealto.com/gpu-sorting.html

namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval>
    struct OrderingPrimitive {

        inline static void _order(Tkey &a, Tkey &b, Tval &va, Tval &vb, bool reverse) {
            bool swap   = reverse ^ (a < b);
            Tkey auxa   = a;
            Tkey auxb   = b;
            Tval auxida = va;
            Tval auxidb = vb;
            a           = (swap) ? auxb : auxa;
            b           = (swap) ? auxa : auxb;
            va          = (swap) ? auxidb : auxida;
            vb          = (swap) ? auxida : auxidb;
        }

        inline static void
        _orderV(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, u32 b, bool reverse) {
            bool swap   = reverse ^ (x[a] < x[b]);
            auto auxa   = x[a];
            auto auxb   = x[b];
            auto auxida = vx[a];
            auto auxidb = vx[b];
            x[a]        = (swap) ? auxb : auxa;
            x[b]        = (swap) ? auxa : auxb;
            vx[a]       = (swap) ? auxidb : auxida;
            vx[b]       = (swap) ? auxida : auxidb;
        }

        template<u32 stencil_size>
        static void order_stencil(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse);

        template<>
        inline void
        order_stencil<2>(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse) {
            _orderV(x, vx, a, a + 1, reverse);
        }

        template<>
        inline void
        order_stencil<4>(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse) {
#pragma unroll
            for (int i4 = 0; i4 < 2; i4++) {
                _orderV(x, vx, a + i4, a + i4 + 2, reverse);
            }
            order_stencil<2>(x, vx, a, reverse);
            order_stencil<2>(x, vx, a + 2, reverse);
        }

        template<>
        inline void
        order_stencil<8>(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse) {
#pragma unroll
            for (int i8 = 0; i8 < 4; i8++) {
                _orderV(x, vx, a + i8, a + i8 + 4, reverse);
            }
            order_stencil<4>(x, vx, a, reverse);
            order_stencil<4>(x, vx, a + 4, reverse);
        }

        template<>
        inline void
        order_stencil<16>(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse) {
#pragma unroll
            for (int i16 = 0; i16 < 8; i16++) {
                _orderV(x, vx, a + i16, a + i16 + 8, reverse);
            }
            order_stencil<8>(x, vx, a, reverse);
            order_stencil<8>(x, vx, a + 8, reverse);
        }

        template<>
        inline void
        order_stencil<32>(Tkey *__restrict__ x, Tval *__restrict__ vx, u32 a, bool reverse) {
#pragma unroll
            for (int i32 = 0; i32 < 16; i32++) {
                _orderV(x, vx, a + i32, a + i32 + 16, reverse);
            }
            order_stencil<16>(x, vx, a, reverse);
            order_stencil<16>(x, vx, a + 16, reverse);
        }

        template<u32 stencil_size>
        static void
        order_kernel(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t);

        template<>
        inline void
        order_kernel<32>(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1U;

            _inc >>= 4;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 5) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[32];
#pragma unroll
            for (int k = 0; k < 32; k++)
                x[k] = m[k * _inc + i];

            Tval idx[32];
#pragma unroll
            for (int k = 0; k < 32; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<32>(x, idx, 0, reverse);

// Store
#pragma unroll
            for (int k = 0; k < 32; k++)
                m[k * _inc + i] = x[k];
#pragma unroll
            for (int k = 0; k < 32; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void
        order_kernel<16>(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t) {

            u32 _inc = inc;
            u32 _dir = length << 1;

            _inc >>= 3;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 4) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[16];
#pragma unroll
            for (int k = 0; k < 16; k++)
                x[k] = m[k * _inc + i];

            Tval idx[16];
#pragma unroll
            for (int k = 0; k < 16; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<16>(x, idx, 0, reverse);

// Store
#pragma unroll
            for (int k = 0; k < 16; k++)
                m[k * _inc + i] = x[k];
#pragma unroll
            for (int k = 0; k < 16; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void
        order_kernel<8>(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1;

            _inc >>= 2;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 3) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[8];
#pragma unroll
            for (int k = 0; k < 8; k++)
                x[k] = m[k * _inc + i];

            Tval idx[8];
#pragma unroll
            for (int k = 0; k < 8; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<8>(x, idx, 0, reverse);

// Store
#pragma unroll
            for (int k = 0; k < 8; k++)
                m[k * _inc + i] = x[k];
#pragma unroll
            for (int k = 0; k < 8; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void
        order_kernel<4>(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1;

            _inc >>= 1;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 2) + low; // insert 00 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x0 = m[0 + i];
            Tkey x1 = m[_inc + i];
            Tkey x2 = m[2 * _inc + i];
            Tkey x3 = m[3 * _inc + i];

            Tval idx0 = id[0 + i];
            Tval idx1 = id[_inc + i];
            Tval idx2 = id[2 * _inc + i];
            Tval idx3 = id[3 * _inc + i];

            // Sort
            _order(x0, x2, idx0, idx2, reverse);
            _order(x1, x3, idx1, idx3, reverse);
            _order(x0, x1, idx0, idx1, reverse);
            _order(x2, x3, idx2, idx3, reverse);

            // Store
            m[0 + i]        = x0;
            m[_inc + i]     = x1;
            m[2 * _inc + i] = x2;
            m[3 * _inc + i] = x3;

            id[0 + i]        = idx0;
            id[_inc + i]     = idx1;
            id[2 * _inc + i] = idx2;
            id[3 * _inc + i] = idx3;
        }

        template<>
        inline void
        order_kernel<2>(Tkey *__restrict__ m, Tval *__restrict__ id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1;

            int low      = t & (_inc - 1);    // low order bits (below INC)
            int i        = (t << 1) - low;    // insert 0 at position INC
            bool reverse = ((_dir & i) == 0); // asc/desc order

            u32 addr_1 = 0 + i;
            u32 addr_2 = _inc + i;

            // Load
            Tkey x0   = m[addr_1];
            Tkey x1   = m[addr_2];
            Tval idx0 = id[addr_1];
            Tval idx1 = id[addr_2];

            // Sort
            _order(x0, x1, idx0, idx1, reverse);

            // Store
            m[addr_1]  = x0;
            m[addr_2]  = x1;
            id[addr_1] = idx0;
            id[addr_2] = idx1;
        }
    };

    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated_usm(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            shambase::throw_with_loc<std::invalid_argument>(
                "this algorithm can only be used with length that are powers of two");
        }

        using B = OrderingPrimitive<Tkey, Tval>;

        for (u32 length = 1; length < len; length <<= 1) {
            u32 inc = length;
            while (inc > 0) {
                // log("inc : %d\n",inc);
                // int ninc = 1;
                u32 ninc = 0;

                // B32 sort kernel is less performant than the B16 because of cache size
                if constexpr (MaxStencilSize >= 32) {
                    if (inc >= 16 && ninc == 0) {
                        ninc                  = 5;
                        unsigned int nThreads = len >> ninc;

                        sham::kernel_call_u64(
                            sched->get_queue(),
                            sham::MultiRef{},
                            sham::MultiRef{buf_key, buf_values},
                            nThreads,
                            [=](u64 gid, Tkey *m, Tval *id) {
                                B::template order_kernel<32>(m, id, inc, length, gid);
                            });
                    }
                }

                if constexpr (MaxStencilSize >= 16) {
                    if (inc >= 8 && ninc == 0) {
                        ninc                  = 4;
                        unsigned int nThreads = len >> ninc;

                        sham::kernel_call_u64(
                            sched->get_queue(),
                            sham::MultiRef{},
                            sham::MultiRef{buf_key, buf_values},
                            nThreads,
                            [=](u64 gid, Tkey *m, Tval *id) {
                                B::template order_kernel<16>(m, id, inc, length, gid);
                            });

                        // sort_kernel_B8(arg_eq,* buf_key->buf,*
                        // particles::buf_ids->buf,inc,length<<1);//.wait();
                    }
                }

                if constexpr (MaxStencilSize >= 8) {
                    // B8
                    if (inc >= 4 && ninc == 0) {
                        ninc                  = 3;
                        unsigned int nThreads = len >> ninc;

                        sham::kernel_call_u64(
                            sched->get_queue(),
                            sham::MultiRef{},
                            sham::MultiRef{buf_key, buf_values},
                            nThreads,
                            [](u64 gid, Tkey *m, Tval *id, u32 inc, u32 length) {
                                B::template order_kernel<8>(m, id, inc, length, gid);
                            },
                            inc,
                            length);

                        // sort_kernel_B8(arg_eq,* buf_key->buf,*
                        // particles::buf_ids->buf,inc,length<<1);//.wait();
                    }
                }

                if constexpr (MaxStencilSize >= 4) {
                    // B4
                    if (inc >= 2 && ninc == 0) {
                        ninc                  = 2;
                        unsigned int nThreads = len >> ninc;

                        sham::kernel_call_u64(
                            sched->get_queue(),
                            sham::MultiRef{},
                            sham::MultiRef{buf_key, buf_values},
                            nThreads,
                            [=](u64 gid, Tkey *m, Tval *id) {
                                B::template order_kernel<4>(m, id, inc, length, gid);
                            });
                    }
                }

                // B2
                if (ninc == 0) {
                    ninc                  = 1;
                    unsigned int nThreads = len >> ninc;

                    sham::kernel_call_u64(
                        sched->get_queue(),
                        sham::MultiRef{},
                        sham::MultiRef{buf_key, buf_values},
                        nThreads,
                        [=](u64 gid, Tkey *m, Tval *id) {
                            B::template order_kernel<2>(m, id, inc, length, gid);
                        });
                }

                inc >>= ninc;
            }
        }
    }

    template void sort_by_key_bitonic_updated_usm<u32, u32, 16>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<u64, u32, 16>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<u32, u32, 8>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<u64, u32, 8>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<u32, u32, 32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<u64, u32, 32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<f32, f32, 32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<f32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<f64, f64, 32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<f64> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<f32, f32, 16>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<f32> &buf_values,
        u32 len);

    template void sort_by_key_bitonic_updated_usm<f64, f64, 16>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<f64> &buf_values,
        u32 len);

} // namespace shamalgs::algorithm::details

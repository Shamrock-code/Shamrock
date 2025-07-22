// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file bitonicSort_updated_xor_swap.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"

// modified from http://www.bealto.com/gpu-sorting.html

namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval>
    struct OrderingPrimitiveXorSwap {

        using AccKey
            = sycl::accessor<Tkey, 1, sycl::access::mode::read_write, sycl::target::device>;
        using AccVal
            = sycl::accessor<Tval, 1, sycl::access::mode::read_write, sycl::target::device>;

        inline static void _order(Tkey &a, Tkey &b, Tval &va, Tval &vb, bool reverse) {
            bool swap = reverse ^ (a < b);
            if (swap) {
                a ^= b;
                b ^= a;
                a ^= b;
                va ^= vb;
                vb ^= va;
                va ^= vb;
            }
        }

        inline static void _orderV(Tkey *x, Tval *vx, u32 a, u32 b, bool reverse) {
            bool swap = reverse ^ (x[a] < x[b]);

            if (swap) {
                x[a] ^= x[b];
                x[b] ^= x[a];
                x[a] ^= x[b];
                vx[a] ^= vx[b];
                vx[b] ^= vx[a];
                vx[a] ^= vx[b];
            }
        }

        template<u32 stencil_size>
        static void order_stencil(Tkey *x, Tval *vx, u32 a, bool reverse);

        template<>
        inline void order_stencil<2>(Tkey *x, Tval *vx, u32 a, bool reverse) {
            _orderV(x, vx, a, a + 1, reverse);
        }

        template<>
        inline void order_stencil<4>(Tkey *x, Tval *vx, u32 a, bool reverse) {
#pragma unroll
            for (int i4 = 0; i4 < 2; i4++) {
                _orderV(x, vx, a + i4, a + i4 + 2, reverse);
            }
            order_stencil<2>(x, vx, a, reverse);
            order_stencil<2>(x, vx, a + 2, reverse);
        }

        template<>
        inline void order_stencil<8>(Tkey *x, Tval *vx, u32 a, bool reverse) {
#pragma unroll
            for (int i8 = 0; i8 < 4; i8++) {
                _orderV(x, vx, a + i8, a + i8 + 4, reverse);
            }
            order_stencil<4>(x, vx, a, reverse);
            order_stencil<4>(x, vx, a + 4, reverse);
        }

        template<>
        inline void order_stencil<16>(Tkey *x, Tval *vx, u32 a, bool reverse) {
#pragma unroll
            for (int i16 = 0; i16 < 8; i16++) {
                _orderV(x, vx, a + i16, a + i16 + 8, reverse);
            }
            order_stencil<8>(x, vx, a, reverse);
            order_stencil<8>(x, vx, a + 8, reverse);
        }

        template<>
        inline void order_stencil<32>(Tkey *x, Tval *vx, u32 a, bool reverse) {
#pragma unroll
            for (int i32 = 0; i32 < 16; i32++) {
                _orderV(x, vx, a + i32, a + i32 + 16, reverse);
            }
            order_stencil<16>(x, vx, a, reverse);
            order_stencil<16>(x, vx, a + 16, reverse);
        }

        template<u32 stencil_size>
        static void order_kernel(AccKey m, AccVal id, u32 inc, u32 length, i32 t);

        template<>
        inline void order_kernel<32>(AccKey m, AccVal id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1U;

            _inc >>= 4;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 5) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[32];
            for (int k = 0; k < 32; k++)
                x[k] = m[k * _inc + i];

            uint idx[32];
            for (int k = 0; k < 32; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<32>(x, idx, 0, reverse);

            // Store
            for (int k = 0; k < 32; k++)
                m[k * _inc + i] = x[k];
            for (int k = 0; k < 32; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void order_kernel<16>(AccKey m, AccVal id, u32 inc, u32 length, i32 t) {

            u32 _inc = inc;
            u32 _dir = length << 1;

            _inc >>= 3;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 4) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[16];
            for (int k = 0; k < 16; k++)
                x[k] = m[k * _inc + i];

            Tval idx[16];
            for (int k = 0; k < 16; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<16>(x, idx, 0, reverse);

            // Store
            for (int k = 0; k < 16; k++)
                m[k * _inc + i] = x[k];
            for (int k = 0; k < 16; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void order_kernel<8>(AccKey m, AccVal id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1;

            _inc >>= 2;
            int low      = t & (_inc - 1);         // low order bits (below INC)
            int i        = ((t - low) << 3) + low; // insert 000 at position INC
            bool reverse = ((_dir & i) == 0);      // asc/desc order

            // Load
            Tkey x[8];
            for (int k = 0; k < 8; k++)
                x[k] = m[k * _inc + i];

            Tval idx[8];
            for (int k = 0; k < 8; k++)
                idx[k] = id[k * _inc + i];

            // Sort
            order_stencil<8>(x, idx, 0, reverse);

            // Store
            for (int k = 0; k < 8; k++)
                m[k * _inc + i] = x[k];
            for (int k = 0; k < 8; k++)
                id[k * _inc + i] = idx[k];
        }

        template<>
        inline void order_kernel<4>(AccKey m, AccVal id, u32 inc, u32 length, i32 t) {
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
        inline void order_kernel<2>(AccKey m, AccVal id, u32 inc, u32 length, i32 t) {
            u32 _inc = inc;
            u32 _dir = length << 1;

            int low      = t & (_inc - 1);    // low order bits (below INC)
            int i        = (t << 1) - low;    // insert 0 at position INC
            bool reverse = ((_dir & i) == 0); // asc/desc order

            // Load
            Tkey x0   = m[0 + i];
            Tkey x1   = m[_inc + i];
            Tval idx0 = id[0 + i];
            Tval idx1 = id[_inc + i];

            // Sort
            _order(x0, x1, idx0, idx1, reverse);

            // Store
            m[0 + i]     = x0;
            m[_inc + i]  = x1;
            id[0 + i]    = idx0;
            id[_inc + i] = idx1;
        }
    };

    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated_xor_swap(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            shambase::throw_with_loc<std::invalid_argument>(
                "this algorithm can only be used with length that are powers of two");
        }

        using B = OrderingPrimitiveXorSwap<Tkey, Tval>;

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
                        sycl::range<1> range{nThreads};

                        auto ker_sort_morton_b32 = [&](sycl::handler &cgh) {
                            sycl::accessor m{buf_key, cgh, sycl::read_write};
                            sycl::accessor id{buf_values, cgh, sycl::read_write};

                            cgh.parallel_for(range, [=](sycl::item<1> item) {
                                //(__global data_t * data,__global uint * ids,int inc,int dir)

                                B::template order_kernel<32>(m, id, inc, length, item.get_id(0));
                            });
                        };
                        q.submit(ker_sort_morton_b32);
                    }
                }

                if constexpr (MaxStencilSize >= 16) {
                    if (inc >= 8 && ninc == 0) {
                        ninc                  = 4;
                        unsigned int nThreads = len >> ninc;
                        sycl::range<1> range{nThreads};

                        auto ker_sort_morton_b16 = [&](sycl::handler &cgh) {
                            sycl::accessor m{buf_key, cgh, sycl::read_write};
                            sycl::accessor id{buf_values, cgh, sycl::read_write};

                            cgh.parallel_for(range, [=](sycl::item<1> item) {
                                //(__global data_t * data,__global uint * ids,int inc,int dir)

                                B::template order_kernel<16>(m, id, inc, length, item.get_id(0));
                            });
                        };
                        q.submit(ker_sort_morton_b16);

                        // sort_kernel_B8(arg_eq,* buf_key->buf,*
                        // particles::buf_ids->buf,inc,length<<1);//.wait();
                    }
                }

                if constexpr (MaxStencilSize >= 8) {
                    // B8
                    if (inc >= 4 && ninc == 0) {
                        ninc                  = 3;
                        unsigned int nThreads = len >> ninc;
                        sycl::range<1> range{nThreads};

                        auto ker_sort_morton_b8 = [&](sycl::handler &cgh) {
                            sycl::accessor m{buf_key, cgh, sycl::read_write};
                            sycl::accessor id{buf_values, cgh, sycl::read_write};

                            cgh.parallel_for(range, [=](sycl::item<1> item) {
                                //(__global data_t * data,__global uint * ids,int inc,int dir)

                                B::template order_kernel<8>(m, id, inc, length, item.get_id(0));
                            });
                        };
                        q.submit(ker_sort_morton_b8);

                        // sort_kernel_B8(arg_eq,* buf_key->buf,*
                        // particles::buf_ids->buf,inc,length<<1);//.wait();
                    }
                }

                if constexpr (MaxStencilSize >= 4) {
                    // B4
                    if (inc >= 2 && ninc == 0) {
                        ninc                  = 2;
                        unsigned int nThreads = len >> ninc;
                        sycl::range<1> range{nThreads};
                        // sort_kernel_B4(arg_eq,* buf_key->buf,*
                        // particles::buf_ids->buf,inc,length<<1);
                        auto ker_sort_morton_b4 = [&](sycl::handler &cgh) {
                            sycl::accessor m{buf_key, cgh, sycl::read_write};
                            sycl::accessor id{buf_values, cgh, sycl::read_write};
                            cgh.parallel_for(range, [=](sycl::item<1> item) {
                                B::template order_kernel<4>(m, id, inc, length, item.get_id(0));
                            });
                        };
                        q.submit(ker_sort_morton_b4);
                    }
                }

                // B2
                if (ninc == 0) {
                    ninc                  = 1;
                    unsigned int nThreads = len >> ninc;
                    sycl::range<1> range{nThreads};
                    // sort_kernel_B2(arg_eq,* buf_key->buf,*
                    // particles::buf_ids->buf,inc,length<<1);
                    auto ker_sort_morton_b2 = [&](sycl::handler &cgh) {
                        sycl::accessor m{buf_key, cgh, sycl::read_write};
                        sycl::accessor id{buf_values, cgh, sycl::read_write};

                        cgh.parallel_for(range, [=](sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            B::template order_kernel<2>(m, id, inc, length, item.get_id(0));
                        });
                    };
                    q.submit(ker_sort_morton_b2);
                }

                inc >>= ninc;
            }
        }
    }

    template void sort_by_key_bitonic_updated_xor_swap<u32, u32, 16>(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_updated_xor_swap<u64, u32, 16>(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_updated_xor_swap<u32, u32, 8>(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_updated_xor_swap<u64, u32, 8>(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_updated_xor_swap<u32, u32, 32>(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_updated_xor_swap<u64, u32, 32>(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

} // namespace shamalgs::algorithm::details

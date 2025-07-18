// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file bitonicSort_legacy.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"
#include "shamcomm/logs.hpp"
#include <stdexcept>

// modified from http://www.bealto.com/gpu-sorting.html

#define MAXORDER_SORT_KERNEL 16

#define ORDER(a, b, ida, idb)                                                                      \
    {                                                                                              \
        bool swap   = reverse ^ (a < b);                                                           \
        Tkey auxa   = a;                                                                           \
        Tkey auxb   = b;                                                                           \
        Tval auxida = ida;                                                                         \
        Tval auxidb = idb;                                                                         \
        a           = (swap) ? auxb : auxa;                                                        \
        b           = (swap) ? auxa : auxb;                                                        \
        ida         = (swap) ? auxidb : auxida;                                                    \
        idb         = (swap) ? auxida : auxidb;                                                    \
    }

#define ORDERV(x, idx, a, b)                                                                       \
    {                                                                                              \
        bool swap   = reverse ^ (x[a] < x[b]);                                                     \
        Tkey auxa   = x[a];                                                                        \
        Tkey auxb   = x[b];                                                                        \
        Tval auxida = idx[a];                                                                      \
        Tval auxidb = idx[b];                                                                      \
        x[a]        = (swap) ? auxb : auxa;                                                        \
        x[b]        = (swap) ? auxa : auxb;                                                        \
        idx[a]      = (swap) ? auxidb : auxida;                                                    \
        idx[b]      = (swap) ? auxida : auxidb;                                                    \
    }

#define B2V(x, idx, a) {ORDERV(x, idx, a, a + 1)}

#define B4V(x, idx, a)                                                                             \
    {                                                                                              \
        for (int i4 = 0; i4 < 2; i4++) {                                                           \
            ORDERV(x, idx, a + i4, a + i4 + 2)                                                     \
        }                                                                                          \
        B2V(x, idx, a) B2V(x, idx, a + 2)                                                          \
    }

#define B8V(x, idx, a)                                                                             \
    {                                                                                              \
        for (int i8 = 0; i8 < 4; i8++) {                                                           \
            ORDERV(x, idx, a + i8, a + i8 + 4)                                                     \
        }                                                                                          \
        B4V(x, idx, a) B4V(x, idx, a + 4)                                                          \
    }

#define B16V(x, idx, a)                                                                            \
    {                                                                                              \
        for (int i16 = 0; i16 < 8; i16++) {                                                        \
            ORDERV(x, idx, a + i16, a + i16 + 8)                                                   \
        }                                                                                          \
        B8V(x, idx, a) B8V(x, idx, a + 8)                                                          \
    }

#define B32V(x, idx, a)                                                                            \
    {                                                                                              \
        for (int i32 = 0; i32 < 16; i32++) {                                                       \
            ORDERV(x, idx, a + i32, a + i32 + 16)                                                  \
        }                                                                                          \
        B16V(x, idx, a) B16V(x, idx, a + 16)                                                       \
    }

class Bitonic_sort_B32_morton32;
class Bitonic_sort_B16_morton32;
class Bitonic_sort_B8_morton32;
class Bitonic_sort_B4_morton32;
class Bitonic_sort_B2_morton32;

class Bitonic_sort_B32_morton64;
class Bitonic_sort_B16_morton64;
class Bitonic_sort_B8_morton64;
class Bitonic_sort_B4_morton64;
class Bitonic_sort_B2_morton64;

namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval>
    void sort_by_key_bitonic_legacy(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            shambase::throw_with_loc<std::invalid_argument>(
                "this algorithm can only be used with length that are powers of two");
        }

        shamcomm::logs::debug_sycl_ln(
            "BitonicSorter", "submit : sycl_sort_morton_key_pair<u32, MultiKernel>");

        for (u32 length = 1; length < len; length <<= 1) {
            u32 inc = length;
            while (inc > 0) {
                // log("inc : %d\n",inc);
                // int ninc = 1;
                u32 ninc = 0;

// B32 sort kernel is less performant than the B16 because of cache size
#if MAXORDER_SORT_KERNEL >= 32
                if (inc >= 16 && ninc == 0) {
                    ninc                  = 5;
                    unsigned int nThreads = len >> ninc;
                    sycl::range<1> range{nThreads};

                    auto ker_sort_morton_b32 = [&](sycl::handler &cgh) {
                        sycl::accessor m{buf_key, cgh, sycl::read_write};
                        sycl::accessor id{buf_values, cgh, sycl::read_write};

                        cgh.parallel_for(range, [=](sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length << 1;

                            _inc >>= 4;
                            int t        = item.get_id();          // thread index
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
                            B32V(x, idx, 0)

                            // Store
                            for (int k = 0; k < 32; k++)
                                m[k * _inc + i] = x[k];
                            for (int k = 0; k < 32; k++)
                                id[k * _inc + i] = idx[k];
                        });
                    };
                    q.submit(ker_sort_morton_b32);
                }
#endif

#if MAXORDER_SORT_KERNEL >= 16
                if (inc >= 8 && ninc == 0) {
                    ninc                  = 4;
                    unsigned int nThreads = len >> ninc;
                    sycl::range<1> range{nThreads};

                    auto ker_sort_morton_b16 = [&](sycl::handler &cgh) {
                        sycl::accessor m{buf_key, cgh, sycl::read_write};
                        sycl::accessor id{buf_values, cgh, sycl::read_write};

                        cgh.parallel_for(range, [=](sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length << 1;

                            _inc >>= 3;
                            int t        = item.get_id(0);         // thread index
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
                            B16V(x, idx, 0)

                            // Store
                            for (int k = 0; k < 16; k++)
                                m[k * _inc + i] = x[k];
                            for (int k = 0; k < 16; k++)
                                id[k * _inc + i] = idx[k];
                        });
                    };
                    q.submit(ker_sort_morton_b16);

                    // sort_kernel_B8(arg_eq,* buf_key->buf,*
                    // particles::buf_ids->buf,inc,length<<1);//.wait();
                }
#endif

#if MAXORDER_SORT_KERNEL >= 8
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

                            u32 _inc = inc;
                            u32 _dir = length << 1;

                            _inc >>= 2;
                            int t        = item.get_id(0);         // thread index
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
                            B8V(x, idx, 0)

                            // Store
                            for (int k = 0; k < 8; k++)
                                m[k * _inc + i] = x[k];
                            for (int k = 0; k < 8; k++)
                                id[k * _inc + i] = idx[k];
                        });
                    };
                    q.submit(ker_sort_morton_b8);

                    // sort_kernel_B8(arg_eq,* buf_key->buf,*
                    // particles::buf_ids->buf,inc,length<<1);//.wait();
                }
#endif

#if MAXORDER_SORT_KERNEL >= 4
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
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length << 1;

                            _inc >>= 1;
                            int t        = item.get_id(0);         // thread index
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
                            ORDER(x0, x2, idx0, idx2)
                            ORDER(x1, x3, idx1, idx3)
                            ORDER(x0, x1, idx0, idx1)
                            ORDER(x2, x3, idx2, idx3)

                            // Store
                            m[0 + i]        = x0;
                            m[_inc + i]     = x1;
                            m[2 * _inc + i] = x2;
                            m[3 * _inc + i] = x3;

                            id[0 + i]        = idx0;
                            id[_inc + i]     = idx1;
                            id[2 * _inc + i] = idx2;
                            id[3 * _inc + i] = idx3;
                        });
                    };
                    q.submit(ker_sort_morton_b4);
                }
#endif

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

                            u32 _inc = inc;
                            u32 _dir = length << 1;

                            int t        = item.get_id(0);    // thread index
                            int low      = t & (_inc - 1);    // low order bits (below INC)
                            int i        = (t << 1) - low;    // insert 0 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order

                            // Load
                            Tkey x0   = m[0 + i];
                            Tkey x1   = m[_inc + i];
                            Tval idx0 = id[0 + i];
                            Tval idx1 = id[_inc + i];

                            // Sort
                            ORDER(x0, x1, idx0, idx1)

                            // Store
                            m[0 + i]     = x0;
                            m[_inc + i]  = x1;
                            id[0 + i]    = idx0;
                            id[_inc + i] = idx1;
                        });
                    };
                    q.submit(ker_sort_morton_b2);
                }

                inc >>= ninc;
            }
        }
    }

    template void sort_by_key_bitonic_legacy(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_bitonic_legacy(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

} // namespace shamalgs::algorithm::details

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
 * @file scanDecoupledLookback.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shamalgs/atomic/DeviceCounter.hpp"
#include "shamalgs/atomic/DynamicIdGenerator.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    class ScanTile {
        public:
        static constexpr T STATE_X = 0;
        static constexpr T STATE_A = 1;
        static constexpr T STATE_P = 2;

        using PackStorage = u64;

        sycl::vec<T, 2> state;

        inline static ScanTile invalid() { return ScanTile{{STATE_X, 0}}; }

        inline bool has_prefix_available() { return state.x() == STATE_P; }

        inline T get_prefix() { return state.y(); }

        inline static ScanTile unpack(PackStorage s) { return ScanTile{sham::unpack32(s)}; }

        inline static PackStorage pack(T a, T b) { return sham::pack32(a, b); }

        inline bool has_no_prefix() { return state.x() != STATE_P; }

        inline bool is_invalid() { return state.x() == STATE_X; }
    };

    class ScanTile30bitint {
        public:
        static constexpr u32 STATE_X = 0;
        static constexpr u32 STATE_A = 1;
        static constexpr u32 STATE_P = 2;

        using PackStorage = u32;

        sycl::vec<u32, 2> state;

        inline static ScanTile30bitint invalid() { return ScanTile30bitint{{STATE_X, 0}}; }

        inline bool has_prefix_available() { return state.x() == STATE_P; }

        inline u32 get_prefix() { return state.y(); }

        inline static ScanTile30bitint unpack(PackStorage s) {

            constexpr u32 mask = (1U << 30U) - 1U;

            return ScanTile30bitint{sycl::vec<u32, 2>{s >> 30U, s & mask}};
        }

        inline static PackStorage pack(u32 a, u32 b) { return (a << 30U) + b; }

        inline bool has_no_prefix() { return state.x() != STATE_P; }

        inline bool is_invalid() { return state.x() == STATE_X; }
    };

    enum DecoupledLoockBackPolicy { Standard, Parralelized };

    template<class T, u32 group_size, DecoupledLoockBackPolicy policy, class Tile>
    class ScanDecoupledLoockBack;

    template<class T, u32 group_size, DecoupledLoockBackPolicy policy, class Tile>
    class ScanDecoupledLoockBackAccessed {
        public:
        sycl::accessor<typename Tile::PackStorage, 1, sycl::access::mode::read_write>
            acc_tile_state;

        sycl::local_accessor<T, 1> local_scan_buf;
        sycl::local_accessor<T, 1> local_sum;

        u32 group_count;

        using atomic_ref_T = sycl::atomic_ref<
            typename Tile::PackStorage,
            sycl::memory_order_relaxed,
            sycl::memory_scope_work_group,
            sycl::access::address_space::global_space>;

        ScanDecoupledLoockBackAccessed(
            sycl::handler &cgh,
            ScanDecoupledLoockBack<T, group_size, policy, Tile> &scan,
            u32 group_count)
            : acc_tile_state{scan.tile_state, cgh, sycl::read_write}, local_scan_buf{1, cgh},
              local_sum{1, cgh}, group_count(group_count) {}

        template<class InputGetter, class OutputSetter>
        inline void decoupled_lookback_scan(
            sycl::nd_item<1> id,
            const u32 local_id,
            const u32 group_tile_id,
            InputGetter input,
            OutputSetter out,
            u32 slice_id = 0) const {

            u32 pointer_offset = slice_id * group_count;

            if (local_id == 0) {

                atomic_ref_T tile_atomic(acc_tile_state[group_tile_id + pointer_offset]);

                // load group sum
                T local_group_sum = input();
                T accum           = 0;
                u32 tile_ptr      = group_tile_id - 1;
                Tile tile_state   = Tile::invalid();

                // global scan using atomic counter

                if (group_tile_id != 0) {

                    tile_atomic.store(Tile::pack(Tile::STATE_A, local_group_sum));

                    while (tile_state.has_no_prefix()) {

                        atomic_ref_T atomic_state(acc_tile_state[tile_ptr + pointer_offset]);

                        do {
                            tile_state = Tile::unpack(atomic_state.load());
                        } while (tile_state.is_invalid());

                        accum += tile_state.get_prefix();

                        tile_ptr--;
                    }
                }

                tile_atomic.store(Tile::pack(Tile::STATE_P, accum + local_group_sum));

                out(accum);
            }

            // sync
            id.barrier(sycl::access::fence_space::local_space);
        }

        inline T scan(
            sycl::nd_item<1> id,
            const u32 local_id,
            const u32 group_tile_id,
            const T input,
            u32 slice_id = 0) const {

            // local scan in the group
            // the local sum will be in local id `group_size - 1`
            T local_scan = sycl::inclusive_scan_over_group(id.get_group(), input, sycl::plus<T>{});

            // can be removed if i change the index in the look back ?
            if (local_id == group_size - 1) {
                local_scan_buf[0] = local_scan;
            }

            // sync group
            id.barrier(sycl::access::fence_space::local_space);

            decoupled_lookback_scan(
                id,
                local_id,
                group_tile_id,
                [=]() {
                    return local_scan_buf[0];
                },
                [=](T accum) {
                    local_sum[0] = accum;
                },
                slice_id);

            return local_scan + local_sum[0];
        }
    };

    template<class T, u32 group_size, DecoupledLoockBackPolicy policy, class Tile>
    class ScanDecoupledLoockBack {
        public:
        u32 slice_count;
        u32 group_count;

        sycl::buffer<typename Tile::PackStorage> tile_state;

        ScanDecoupledLoockBack(sycl::queue &q, u32 group_count, u32 slice_count = 1)
            : slice_count(slice_count), group_count(group_count),
              tile_state(group_count * slice_count) {

            shamalgs::memory::buf_fill_discard(q, tile_state, Tile::pack(Tile::STATE_X, T(0)));
        }

        using atomic_ref_T = sycl::atomic_ref<
            typename Tile::PackStorage,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        inline ScanDecoupledLoockBackAccessed<T, group_size, policy, Tile>
        get_access(sycl::handler &cgh) {
            return ScanDecoupledLoockBackAccessed<T, group_size, policy, Tile>{
                cgh, *this, group_count};
        }
    };

    template<class T, u32 group_size>
    class InplaceExclusiveScanDecoupledLookBack;

    template<class T, u32 group_size>
    void
    exclusive_sum_in_place_atomic_decoupled_v5(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // group aggregates
        ScanDecoupledLoockBack<T, group_size, Standard, ScanTile<T>> dlookbackscan(q, group_cnt);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_value{buf1, cgh, sycl::read_write};

            auto scanop = dlookbackscan.get_access(cgh);

            cgh.parallel_for<InplaceExclusiveScanDecoupledLookBack<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id      = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 global_id     = group_tile_id * group_size + local_id;

                    // load from global buffer
                    T local_val = (global_id > 0 && global_id < len) ? acc_value[global_id - 1] : 0;

                    T scanned_value = scanop.scan(id, local_id, group_tile_id, local_val);

                    // store final result
                    if (global_id < len) {
                        acc_value[global_id] = scanned_value;
                    }
                });
        });
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v5;

    template<class T, u32 group_size>
    sycl::buffer<T>
    exclusive_sum_atomic_decoupled_v5(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(corrected_len);

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<typename ScanTile<T>::PackStorage> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            auto dyn_id = id_gen.get_access(cgh);

            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            using atomic_ref_T = sycl::atomic_ref<
                u64,
                sycl::memory_order_relaxed,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v5<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id = id.get_local_id(0);

                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    u32 group_tile_id = group_id.dyn_group_id;
                    u32 global_id     = group_id.dyn_global_id;
                    // u32 group_tile_id = id.get_group_linear_id();
                    // u32 global_id = group_tile_id * group_size + local_id;

                    // load from global buffer
                    T local_val = (global_id > 0 && global_id < len) ? acc_in[global_id - 1] : 0;

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    // can be removed if i change the index in the look back ?
                    if (local_id == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (local_id == 0) {

                        atomic_ref_T tile_atomic(acc_tile_state[group_tile_id]);

                        // load group sum
                        T local_group_sum          = local_scan_buf[0];
                        T accum                    = 0;
                        u32 tile_ptr               = group_tile_id - 1;
                        sycl::vec<T, 2> tile_state = {STATE_X, 0};

                        // global scan using atomic counter

                        if (group_tile_id != 0) {

                            tile_atomic.store(sham::pack32(STATE_A, local_group_sum));

                            while (tile_state.x() != STATE_P) {

                                atomic_ref_T atomic_state(acc_tile_state[tile_ptr]);

                                do {
                                    tile_state = sham::unpack32(atomic_state.load());
                                } while (tile_state.x() == STATE_X);

                                accum += tile_state.y();

                                tile_ptr--;
                            }
                        }

                        tile_atomic.store(sham::pack32(STATE_P, accum + local_group_sum));

                        local_sum[0] = accum;
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (global_id < len) {
                        acc_out[global_id] = local_scan + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v5_USM;

    template<class T, u32 group_size>
    sham::DeviceBuffer<T> exclusive_sum_atomic_decoupled_v5_usm(
        sham::DeviceScheduler_ptr dev_sched, sham::DeviceBuffer<T, sham::device> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sham::DeviceBuffer<T> ret_buf(corrected_len, dev_sched);

        // group aggregates
        sycl::buffer<typename ScanTile<T>::PackStorage> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(
            dev_sched->get_queue().q, tile_state, sham::pack32(STATE_X, T(0)));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(dev_sched->get_queue().q);

        sham::EventList depends_list;
        const T *in_ptr = buf1.get_read_access(depends_list);
        T *out_ptr      = ret_buf.get_write_access(depends_list);

        sycl::event e = dev_sched->get_queue().submit(
            depends_list, [&, group_cnt, len, in_ptr, out_ptr](sycl::handler &cgh) {
                auto dyn_id = id_gen.get_access(cgh);

                sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

                sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
                sycl::local_accessor<T, 1> local_sum{1, cgh};

                using atomic_ref_T = sycl::atomic_ref<
                    u64,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>;

                cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v5_USM<T, group_size>>(
                    sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                        u32 local_id = id.get_local_id(0);

                        atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                        u32 group_tile_id = group_id.dyn_group_id;
                        u32 global_id     = group_id.dyn_global_id;
                        // u32 group_tile_id = id.get_group_linear_id();
                        // u32 global_id = group_tile_id * group_size + local_id;

                        // load from global buffer
                        T local_val
                            = (global_id > 0 && global_id < len) ? in_ptr[global_id - 1] : 0;

                        // local scan in the group
                        // the local sum will be in local id `group_size - 1`
                        T local_scan = sycl::inclusive_scan_over_group(
                            id.get_group(), local_val, sycl::plus<T>{});

                        // can be removed if i change the index in the look back ?
                        if (local_id == group_size - 1) {
                            local_scan_buf[0] = local_scan;
                        }

                        // sync group
                        id.barrier(sycl::access::fence_space::local_space);

                        // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                        if (local_id == 0) {

                            atomic_ref_T tile_atomic(acc_tile_state[group_tile_id]);

                            // load group sum
                            T local_group_sum          = local_scan_buf[0];
                            T accum                    = 0;
                            u32 tile_ptr               = group_tile_id - 1;
                            sycl::vec<T, 2> tile_state = {STATE_X, 0};

                            // global scan using atomic counter

                            if (group_tile_id != 0) {

                                tile_atomic.store(sham::pack32(STATE_A, local_group_sum));

                                while (tile_state.x() != STATE_P) {

                                    atomic_ref_T atomic_state(acc_tile_state[tile_ptr]);

                                    do {
                                        tile_state = sham::unpack32(atomic_state.load());
                                    } while (tile_state.x() == STATE_X);

                                    accum += tile_state.y();

                                    tile_ptr--;
                                }
                            }

                            tile_atomic.store(sham::pack32(STATE_P, accum + local_group_sum));

                            local_sum[0] = accum;
                        }

                        // sync
                        id.barrier(sycl::access::fence_space::local_space);

                        // store final result
                        if (global_id < len) {
                            out_ptr[global_id] = local_scan + local_sum[0];
                        }
                    });
            });
        buf1.complete_event_state(e);
        ret_buf.complete_event_state(e);

        // Without this the returned buffer is wrong
        ret_buf.resize(len);

        return ret_buf;
    }

    template<class T, u32 group_size, u32 thread_counts>
    class KernelExclusiveSumAtomicSyncDecoupled_v6;

    template<class T, u32 group_size, u32 thread_counts>
    sycl::buffer<T>
    exclusive_sum_atomic_decoupled_v6(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(corrected_len);

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<typename ScanTile<T>::PackStorage> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            // sycl::stream dump (4096, 1024, cgh);

            using atomic_ref_T = sycl::atomic_ref<
                u64,
                sycl::memory_order_relaxed,
                sycl::memory_scope_work_group,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<
                KernelExclusiveSumAtomicSyncDecoupled_v6<T, group_size, thread_counts>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id      = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 global_id     = group_tile_id * group_size + local_id;

                    auto local_group = id.get_group();

                    // load from global buffer
                    T local_val = (global_id > 0 && global_id < len) ? acc_in[global_id - 1] : 0;
                    ;

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan
                        = sycl::inclusive_scan_over_group(local_group, local_val, sycl::plus<T>{});

                    if (local_id == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    // parralelized lookback
                    static_assert(thread_counts <= group_size, "impossible");

                    T local_group_sum = local_scan_buf[0];
                    T accum           = 0;

                    T sum_state;
                    u32 last_p_index;

                    if (group_tile_id != 0) {
                        if (local_id == 0) {
                            atomic_ref_T(acc_tile_state[group_tile_id])
                                .store(sham::pack32(STATE_A, local_group_sum));
                        }

                        sycl::vec<T, 2> tile_state;
                        u32 group_tile_ptr = group_tile_id - 1;

                        bool continue_loop = true;

                        do {

                            if ((local_id < thread_counts) && (group_tile_ptr >= local_id)) {
                                atomic_ref_T atomic_state(
                                    acc_tile_state[group_tile_ptr - local_id]);

                                do {
                                    tile_state = sham::unpack32(atomic_state.load());
                                } while (tile_state.x() == STATE_X);

                            } else {
                                tile_state = {STATE_A, 0};
                            }

                            // if(group_tile_id == 25) dump << "ps : " << tile_state << "\n";

                            sum_state = sycl::reduce_over_group(
                                local_group, tile_state.x(), sycl::plus<T>{});

                            // if(group_tile_id == 25) dump << "ss : " << sum_state << "\n";

                            if (sum_state > group_size) {
                                // there is a P

                                continue_loop = false;

                                last_p_index = sycl::reduce_over_group(
                                    local_group,
                                    (tile_state.x() == STATE_P) ? (local_id) : (group_size),
                                    sycl::minimum<T>{});

                                // if(group_tile_id == 25) dump << "lp : " << last_p_index << "\n";

                                tile_state.y() = (local_id <= last_p_index) ? tile_state.y() : 0;

                                // if(group_tile_id == 25) dump << "ts : " << tile_state << "\n";

                            } else {
                                // there is only A's
                                continue_loop = (group_tile_ptr >= thread_counts);
                                group_tile_ptr -= thread_counts;
                            }

                            accum += sycl::reduce_over_group(
                                local_group, tile_state.y(), sycl::plus<T>{});

                            // if(group_tile_id == 25) dump << "as : " << accum << "\n";

                        } while (continue_loop);
                    }

                    if (local_id == 0) {
                        atomic_ref_T(acc_tile_state[group_tile_id])
                            .store(sham::pack32(STATE_P, accum + local_group_sum));
                    }

                    // store final result
                    if (global_id < len) {
                        acc_out[global_id] = accum + local_scan;
                    }
                });
        });

        return ret_buf;
    }

} // namespace shamalgs::numeric::details

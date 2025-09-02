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
 * @file exclusiveScanAtomic.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shamalgs/atomic/DeviceCounter.hpp"
#include "shamalgs/atomic/DynamicIdGenerator.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSync;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic2pass(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt     = shambase::group_count(len, group_size);
        u32 corrected_len = group_cnt * group_size;
        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        atomic::DeviceCounter<i32> device_count(q);
        atomic::DeviceCounter<u32> global_summation(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor value_buffer{ret_buf, cgh, sycl::read_write};

            auto dyn_id         = id_gen.get_access(cgh);
            auto device_counter = device_count.get_access(cgh);
            auto global_sum     = global_summation.get_access(cgh);

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            cgh.parallel_for<KernelExclusiveSumAtomicSync<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val;
                    if (group_id.dyn_global_id < len) {
                        local_val = value_buffer[group_id.dyn_global_id];
                    } else {
                        local_val = 0;
                    }

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // setup device counter atomic
                        sycl::atomic_ref atomic_counter
                            = device_counter.attach_atomic<sycl::memory_order_acq_rel>();
                        sycl::atomic_ref atomic_sum
                            = global_sum.attach_atomic<sycl::memory_order_relaxed>();

                        // load group sum
                        T group_sum = local_scan_buf[0];

                        // global scan using atomic counter

                        if (group_id.dyn_group_id == 0) {

                            // store local sum
                            atomic_sum += group_sum;
                            atomic_counter++;
                            local_sum[0] = 0;

                        } else {
                            while (atomic_counter.load() != group_id.dyn_group_id) {
                            }

                            T exclusive_group_prefix_sum = atomic_sum.fetch_add(group_sum);

                            atomic_counter++;
                            local_sum[0] = exclusive_group_prefix_sum;
                        }
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (group_id.dyn_global_id < len) {
                        value_buffer[group_id.dyn_global_id] = local_scan + local_sum[0];
                        // local_scan - local_val + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSync_v2;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic2pass_v2(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt     = shambase::group_count(len, group_size);
        u32 corrected_len = group_cnt * group_size;
        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<T> aggregates(group_cnt);

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        atomic::DeviceCounter<i32> device_count(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor value_buffer{ret_buf, cgh, sycl::read_write};

            auto dyn_id         = id_gen.get_access(cgh);
            auto device_counter = device_count.get_access(cgh);

            sycl::accessor acc_gsum{aggregates, cgh, sycl::read_write};

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            cgh.parallel_for<KernelExclusiveSumAtomicSync_v2<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val;
                    if (group_id.dyn_global_id < len) {
                        local_val = value_buffer[group_id.dyn_global_id];
                    } else {
                        local_val = 0;
                    }

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // setup device counter atomic
                        sycl::atomic_ref atomic_counter
                            = device_counter.attach_atomic<sycl::memory_order_acq_rel>();

                        // load group sum
                        T group_sum = local_scan_buf[0];

                        // global scan using atomic counter

                        using atomic_ref_T = sycl::atomic_ref<
                            T,
                            sycl::memory_order_relaxed,
                            sycl::memory_scope_device,
                            sycl::access::address_space::global_space>;

                        if (group_id.dyn_group_id == 0) {

                            // store local sum
                            atomic_ref_T(acc_gsum[0]).store(group_sum);

                            atomic_counter++;
                            local_sum[0] = 0;

                        } else {
                            while (atomic_counter.load() != group_id.dyn_group_id) {
                            }

                            T exclusive_group_prefix_sum
                                = atomic_ref_T(acc_gsum[group_id.dyn_group_id - 1]).load();

                            atomic_ref_T(acc_gsum[group_id.dyn_group_id])
                                .store(exclusive_group_prefix_sum + group_sum);

                            atomic_counter++;
                            local_sum[0] = exclusive_group_prefix_sum;
                        }
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (group_id.dyn_global_id < len) {
                        value_buffer[group_id.dyn_global_id] = local_scan + local_sum[0];
                        // local_scan - local_val + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic_decoupled(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt     = shambase::group_count(len, group_size);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<i32> tile_state(group_cnt);
        sycl::buffer<T> tile_aggregates(group_cnt);
        sycl::buffer<T> tile_incl_prefix(group_cnt);

        constexpr i32 STATE_X = 0;
        constexpr i32 STATE_A = 1;
        constexpr i32 STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, STATE_X);
        shamalgs::memory::buf_fill_discard(q, tile_aggregates, T(0));
        shamalgs::memory::buf_fill_discard(q, tile_incl_prefix, T(0));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_value{ret_buf, cgh, sycl::read_write};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};
            sycl::accessor acc_tile_aggregates{tile_aggregates, cgh, sycl::read_write};
            sycl::accessor acc_tile_incl_prefix{tile_incl_prefix, cgh, sycl::read_write};

            auto dyn_id = id_gen.get_access(cgh);

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            using atomic_ref_state = sycl::atomic_ref<
                i32,
                sycl::memory_order_relaxed,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>;

            using atomic_ref_T = sycl::atomic_ref<
                T,
                sycl::memory_order_relaxed,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val;
                    if (group_id.dyn_global_id < len) {
                        local_val = acc_value[group_id.dyn_global_id];
                    } else {
                        local_val = 0;
                    }

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // load group sum
                        T local_group_sum = local_scan_buf[0];
                        T accum           = 0;

                        // global scan using atomic counter

                        if (group_id.dyn_group_id != 0) {

                            atomic_ref_T(acc_tile_aggregates[group_id.dyn_group_id])
                                .store(local_group_sum);
                            atomic_ref_state(acc_tile_state[group_id.dyn_group_id]).store(STATE_A);

                            u32 tile_ptr = group_id.dyn_group_id - 1;

                            while (true) {
                                i32 tstate = atomic_ref_state(acc_tile_state[tile_ptr]).load();

                                if (tstate == STATE_X) {
                                    continue;
                                }

                                if (tstate == STATE_A) {
                                    accum += atomic_ref_T(acc_tile_aggregates[tile_ptr]).load();
                                }

                                if (tstate == STATE_P) {
                                    accum += atomic_ref_T(acc_tile_incl_prefix[tile_ptr]).load();
                                    break;
                                }

                                tile_ptr--;
                            }
                        }

                        atomic_ref_T(acc_tile_incl_prefix[group_id.dyn_group_id])
                            .store(accum + local_group_sum);
                        atomic_ref_state(acc_tile_state[group_id.dyn_group_id]).store(STATE_P);

                        local_sum[0] = accum;
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (group_id.dyn_global_id < len) {
                        acc_value[group_id.dyn_global_id] = local_scan + local_sum[0];
                        // local_scan - local_val + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v2;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic_decoupled_v2(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt     = shambase::group_count(len, group_size);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<u64> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_value{ret_buf, cgh, sycl::read_write};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

            auto dyn_id = id_gen.get_access(cgh);

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            using atomic_ref_T = sycl::atomic_ref<
                u64,
                sycl::memory_order_relaxed,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v2<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val;
                    if (group_id.dyn_global_id < len) {
                        local_val = acc_value[group_id.dyn_global_id];
                    } else {
                        local_val = 0;
                    }

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    auto store = [=](u32 id, T state, T val) {
                        atomic_ref_T(acc_tile_state[id]).store(sham::pack32(state, val));
                    };

                    auto load = [=](u32 id) -> sycl::vec<T, 2> {
                        return sham::unpack32(atomic_ref_T(acc_tile_state[id]).load());
                    };

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // load group sum
                        T local_group_sum = local_scan_buf[0];
                        T accum           = 0;
                        u32 tile_ptr      = group_id.dyn_group_id - 1;

                        // global scan using atomic counter

                        if (group_id.dyn_group_id != 0) {

                            store(group_id.dyn_group_id, STATE_A, local_group_sum);

                            while (true) {

                                sycl::vec<T, 2> state = load(tile_ptr);

                                if (state.x() == STATE_X) {
                                    continue;
                                }

                                if (state.x() == STATE_A) {
                                    accum += state.y();
                                }

                                if (state.x() == STATE_P) {
                                    accum += state.y();
                                    break;
                                }

                                tile_ptr--;
                            }
                        }

                        store(group_id.dyn_group_id, STATE_P, accum + local_group_sum);

                        local_sum[0] = accum;
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (group_id.dyn_global_id < len) {
                        acc_value[group_id.dyn_global_id] = local_scan + local_sum[0];
                        // local_scan - local_val + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v3;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic_decoupled_v3(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt     = shambase::group_count(len, group_size);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<u64> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_value{ret_buf, cgh, sycl::read_write};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

            auto dyn_id = id_gen.get_access(cgh);

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            using atomic_ref_T = sycl::atomic_ref<
                u64,
                sycl::memory_order_relaxed,
                sycl::memory_scope_work_group,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v3<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val;
                    if (group_id.dyn_global_id < len) {
                        local_val = acc_value[group_id.dyn_global_id];
                    } else {
                        local_val = 0;
                    }

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    auto store = [=](u32 id, T state, T val) {
                        atomic_ref_T(acc_tile_state[id]).store(sham::pack32(state, val));
                    };

                    auto load = [=](u32 id) -> sycl::vec<T, 2> {
                        return sham::unpack32(atomic_ref_T(acc_tile_state[id]).load());
                    };

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // load group sum
                        T local_group_sum = local_scan_buf[0];
                        T accum           = 0;
                        u32 tile_ptr      = group_id.dyn_group_id - 1;

                        // global scan using atomic counter

                        if (group_id.dyn_group_id != 0) {

                            store(group_id.dyn_group_id, STATE_A, local_group_sum);

                            while (true) {

                                sycl::vec<T, 2> state = load(tile_ptr);

                                if (state.x() == STATE_X) {
                                    continue;
                                }

                                if (state.x() == STATE_A) {
                                    accum += state.y();
                                }

                                if (state.x() == STATE_P) {
                                    accum += state.y();
                                    break;
                                }

                                tile_ptr--;
                            }
                        }

                        store(group_id.dyn_group_id, STATE_P, accum + local_group_sum);

                        local_sum[0] = accum;
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    if (group_id.dyn_global_id < len) {
                        acc_value[group_id.dyn_global_id] = local_scan + local_sum[0];
                        // local_scan - local_val + local_sum[0];
                    }
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusiveSumAtomicSyncDecoupled_v4;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_atomic_decoupled_v4(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(corrected_len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{corrected_len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0 && thid < len) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<u64> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_value{ret_buf, cgh, sycl::read_write};
            sycl::accessor acc_tile_state{tile_state, cgh, sycl::read_write};

            auto dyn_id = id_gen.get_access(cgh);

            sycl::local_accessor<T, 1> local_scan_buf{1, cgh};
            sycl::local_accessor<T, 1> local_sum{1, cgh};

            using atomic_ref_T = sycl::atomic_ref<
                u64,
                sycl::memory_order_relaxed,
                sycl::memory_scope_work_group,
                sycl::access::address_space::global_space>;

            cgh.parallel_for<KernelExclusiveSumAtomicSyncDecoupled_v4<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    // load from global buffer
                    T local_val = acc_value[group_id.dyn_global_id];

                    // local scan in the group
                    // the local sum will be in local id `group_size - 1`
                    T local_scan = sycl::inclusive_scan_over_group(
                        id.get_group(), local_val, sycl::plus<T>{});

                    if (id.get_local_id(0) == group_size - 1) {
                        local_scan_buf[0] = local_scan;
                    }

                    // sync group
                    id.barrier(sycl::access::fence_space::local_space);

                    auto store = [=](u32 id, T state, T val) {
                        atomic_ref_T(acc_tile_state[id]).store(sham::pack32(state, val));
                    };

                    auto load = [=](u32 id) -> sycl::vec<T, 2> {
                        return sham::unpack32(atomic_ref_T(acc_tile_state[id]).load());
                    };

                    // DATA PARALLEL C++: MASTERING DPC++ ... device wide synchro
                    if (group_id.is_main_thread) {

                        // load group sum
                        T local_group_sum          = local_scan_buf[0];
                        T accum                    = 0;
                        u32 tile_ptr               = group_id.dyn_group_id - 1;
                        sycl::vec<T, 2> tile_state = {STATE_X, 0};

                        // global scan using atomic counter

                        if (group_id.dyn_group_id != 0) {

                            store(group_id.dyn_group_id, STATE_A, local_group_sum);

                            while (tile_state.x() != STATE_P) {

                                atomic_ref_T atomic_state(acc_tile_state[tile_ptr]);

                                do {
                                    tile_state = sham::unpack32(atomic_state.load());
                                } while (tile_state.x() == STATE_X);

                                accum += tile_state.y();

                                tile_ptr--;
                            }
                        }

                        store(group_id.dyn_group_id, STATE_P, accum + local_group_sum);

                        local_sum[0] = accum;
                    }

                    // sync
                    id.barrier(sycl::access::fence_space::local_space);

                    // store final result
                    acc_value[group_id.dyn_global_id] = local_scan + local_sum[0];
                });
        });

        return ret_buf;
    }

    template<class T, u32 group_size>
    class KernelExclusivesum_sycl_jointalg;

    template<class T, u32 group_size>
    sycl::buffer<T> exclusive_sum_sycl_jointalg(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 group_cnt = shambase::group_count(len, group_size);

        group_cnt         = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // prepare the return buffer by shifting values for the exclusive sum
        sycl::buffer<T> ret_buf(corrected_len);
        sycl::buffer<T> ret_buf2(corrected_len);

        q.submit([&, len](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{ret_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{corrected_len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0 && thid < len) ? acc_in[thid - 1] : 0;
            });
        });

        // logger::raw_ln("shifted : ");
        // shamalgs::memory::print_buf(ret_buf, len, 16,"{:4} ");

        // group aggregates
        sycl::buffer<u64> tile_state(group_cnt);

        constexpr T STATE_X = 0;
        constexpr T STATE_A = 1;
        constexpr T STATE_P = 2;

        shamalgs::memory::buf_fill_discard(q, tile_state, sham::pack32(STATE_X, T(0)));

        q.submit([&, group_cnt, len](sycl::handler &cgh) {
            sycl::accessor acc_in{ret_buf, cgh, sycl::read_write};
            sycl::accessor acc_out{ret_buf2, cgh, sycl::read_write};

            cgh.parallel_for<KernelExclusivesum_sycl_jointalg<T, group_size>>(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    T *first = acc_in.get_pointer();
                    T *last  = first + acc_in.size();

                    T *first_out = acc_out.get_pointer();

                    T excl_val;
                    sycl::joint_inclusive_scan(
                        id.get_group(), first, last, first_out, sycl::plus<T>{});
                });
        });

        return ret_buf2;
    }

} // namespace shamalgs::numeric::details

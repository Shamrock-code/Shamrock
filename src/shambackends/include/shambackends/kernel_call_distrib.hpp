// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file kernel_call_distrib.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/kernel_call.hpp"

namespace sham {

    /**
     * @brief A variant of `sham::MultiRef` for distributed data.
     *
     * This class is a drop-in replacement for `sham::MultiRef` but adapted to work
     * with distributed data.
     *
     * @see sham::MultiRef
     */
    template<class... Targ>
    struct DDMultiRef {
        /// A tuple of references to the buffers.
        using storage_t = std::tuple<Targ &...>;

        /// A tuple of references to the buffers.
        storage_t storage;

        /// Constructor
        DDMultiRef(Targ &...arg) : storage(arg...) {}

        /**
         * @brief Get a MultiRef at a given id.
         *
         * This function returns a MultiRef of the buffers at the given id.
         *
         * @param id The id of the patch for which to get the MultiRef.
         * @returns A MultiRef of the buffers at the given id.
         */
        auto get(u64 id) {
            shamlog_debug_ln(
                "kern call",
                "called DDMultiRef.get, id =",
                id,
                SourceLocation{}.format_one_line_func());
            return std::apply(
                [id](auto &...args) {
                    return sham::MultiRef{args.get(id)...};
                },
                storage);
        }
    };

    /**
     * @brief A variant of `sham::kernel_call` for distributed data.
     *
     * This function is a drop-in replacement for `sham::kernel_call` but adapted to work
     * with distributed data. It is implemented on top of the `sham::kernel_call` infrastructure.
     *
     * @see sham::kernel_call
     * @param dev_sched The scheduler to use to launch the kernel.
     * @param in The input distributed data.
     * @param in_out The input/output distributed data.
     * @param thread_counts The number of threads to use for each patch.
     * @param func The function to call.
     */
    template<class index_t, class RefIn, class RefOut, class Functor>
    inline void distributed_data_kernel_call(
        sham::DeviceScheduler_ptr dev_sched,
        RefIn in,
        RefOut in_out,
        const shambase::DistributedData<index_t> &thread_counts,
        Functor &&func) {

        auto mrefs_in
            = thread_counts.template map<decltype(in.get(0))>([&](u64 id, const index_t &n) {
                  shamlog_debug_ln("kern call", "build multi ref in for patch", id);
                  return in.get(id);
              });

        auto mrefs_in_out
            = thread_counts.template map<decltype(in_out.get(0))>([&](u64 id, const index_t &n) {
                  shamlog_debug_ln("kern call", "build multi ref in_out for patch", id);
                  return in_out.get(id);
              });

        thread_counts.for_each([&](u64 id, const index_t &n) {
            shamlog_debug_ln(
                "kern call", "calling sham::kernel_call on patch", id, " thread count", n);
            sham::kernel_call(
                dev_sched->get_queue(),
                mrefs_in.get(id),
                mrefs_in_out.get(id),
                n,
                std::forward<Functor>(func));
        });
    }

    // version where one supplies a kernel generator in the form of [&](sycl::handler &cgh) { ... }
    template<class index_t, class RefIn, class RefOut, class Functor>
    inline void distributed_data_kernel_call_hndl(
        sham::DeviceScheduler_ptr dev_sched,
        RefIn in,
        RefOut in_out,
        const shambase::DistributedData<index_t> &thread_counts,
        Functor &&kernel_gen) {

        auto mrefs_in
            = thread_counts.template map<decltype(in.get(0))>([&](u64 id, const index_t &n) {
                  shamlog_debug_ln("kern call", "build multi ref in for patch", id);
                  return in.get(id);
              });

        auto mrefs_in_out
            = thread_counts.template map<decltype(in_out.get(0))>([&](u64 id, const index_t &n) {
                  shamlog_debug_ln("kern call", "build multi ref in_out for patch", id);
                  return in_out.get(id);
              });

        thread_counts.for_each([&](u64 id, const index_t &n) {
            shamlog_debug_ln(
                "kern call", "calling sham::kernel_call_hndl on patch", id, " thread count", n);

            sham::kernel_call_hndl(
                dev_sched->get_queue(),
                mrefs_in.get(id),
                mrefs_in_out.get(id),
                n,
                std::forward<Functor>(kernel_gen));
        });
    }

} // namespace sham

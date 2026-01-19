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
 * @file DynamicIdGenerator.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::atomic {

    /**
     * @brief Sycl utility to dynamically generate group ids
     *
     * The goal is to affect each worker with a unique id in growing order,
     * e.g. worker group 2 can not start if worker group 1 is not started
     * The performance overhead is minimal (10^-11 s/element on A100)
     *
     * \todo add figure for overhead measurment
     *
     * Exemple :
     *
     * \code{.cpp}
     * DynamicIdGenerator<i32, group_size> id_gen(q);
     * q.submit([&](sycl::handler &cgh) {
     *
     *    auto dyn_id = id_gen.get_access(cgh);
     *
     *    cgh.parallel_for(sycl::nd_range<1>{len, group_size},
     *        [=](sycl::nd_item<1> id) {
     *
     *            atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);
     *
     *            u32 group_tile_id = group_id.dyn_group_id;
     *            u32 global_id = group_id.dyn_global_id;
     *
     *            // kernel execution
     *
     *        });
     *
     * });
     * \endcode
     *
     * @tparam int_t the int type used by the counter (preferentially u32 or u64)
     * @tparam group_size the group size used in `SYCL`
     */
    template<class int_t, u32 group_size>
    class DynamicIdGenerator;

    /**
     * @brief Object returned by `DynamicIdGenerator` containing information about the worker
     * affected id
     *
     * @tparam int_t
     */
    template<class int_t>
    class DynamicId {
        public:
        int_t is_main_thread;
        int_t dyn_group_id;
        int_t dyn_global_id;
    };

    /**
     * @brief Accesses version of `DynamicIdGenerator` see doc for exemple (`DynamicIdGenerator`)
     *
     * @tparam int_t the int type used by the counter (preferentially u32 or u64)
     * @tparam group_size the group size used in `SYCL`
     */
    template<class int_t, u32 group_size>
    class AccessedDynamicIdGenerator {
        public:
        sycl::accessor<int_t, 1, sycl::access::mode::read_write, sycl::access::target::device>
            group_id;

        sycl::local_accessor<int_t, 1> local_group_id;

        inline AccessedDynamicIdGenerator(
            sycl::handler &cgh, DynamicIdGenerator<int_t, group_size> &gen)
            : group_id{gen.group_id, cgh, sycl::read_write}, local_group_id(1, cgh) {}

        /**
         * @brief compute the local ids and return the result `DynamicId`
         *
         * @param it the nd_item given by SYCL
         * @return DynamicId<int_t> the dynamic id
         */
        inline DynamicId<int_t> compute_id(sycl::nd_item<1> it) const {
            DynamicId<int_t> ret;

            ret.is_main_thread = it.get_local_id(0) == 0 ? 1 : 0;

            if (ret.is_main_thread) {

                sycl::atomic_ref<
                    int_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    atomic_group_id(group_id[0]);

                ret.dyn_group_id  = atomic_group_id.fetch_add(1);
                local_group_id[0] = ret.dyn_group_id;
            }
            it.barrier(sycl::access::fence_space::local_space);
            ret.dyn_group_id = local_group_id[0];

            ret.dyn_global_id = ret.dyn_group_id * group_size + it.get_local_id(0);

            return ret;
        }
    };

    template<class int_t, u32 group_size>
    class DynamicIdGenerator {
        public:
        /**
         * @brief the buffer used for group_id synchronization
         *
         */
        sycl::buffer<int_t> group_id;

        /**
         * @brief Construct `DynamicIdGenerator`
         *
         * @param q the `SYCL` queue
         */
        inline explicit DynamicIdGenerator(sycl::queue &q) : group_id(1) {
            memory::buf_fill_discard(q, group_id, 0);
        }

        /**
         * @brief Get the access to `DynamicIdGenerator` returning the accessed variants
         * `AccessedDynamicIdGenerator`
         *
         * @param cgh the `SYCL` command group handler
         * @return AccessedDynamicIdGenerator<int_t, group_size> the accessed `DynamicIdGenerator`
         */
        inline AccessedDynamicIdGenerator<int_t, group_size> get_access(sycl::handler &cgh) {
            return {cgh, *this};
        }
    };

} // namespace shamalgs::atomic

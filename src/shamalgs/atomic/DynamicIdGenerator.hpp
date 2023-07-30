// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/memory.hpp"
#include "shambase/integer.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs::atomic {

    /**
     * @brief Sycl utility to dynamically generate group ids
     * 
     * @tparam int_t 
     * @tparam group_size 
     */
    template<class int_t, u32 group_size>
    class DynamicIdGenerator;

    template<class int_t>
    class DynamicId {
        public:
        int_t is_main_thread;
        int_t dyn_group_id;
        int_t dyn_global_id;
    };

    template<class int_t, u32 group_size>
    class AccessedDynamicIdGenerator {
        public:
        sycl::accessor<int_t, 1, sycl::access::mode::read_write, sycl::access::target::device>
            group_id;

        sycl::local_accessor<int_t, 1> local_group_id;

        inline AccessedDynamicIdGenerator(
            sycl::handler &cgh, DynamicIdGenerator<int_t, group_size> &gen
        )
            : group_id{gen.group_id, cgh, sycl::read_write}, local_group_id(1, cgh) {}

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
        sycl::buffer<int_t> group_id;

        inline explicit DynamicIdGenerator(sycl::queue &q) : group_id(1) {
            memory::buf_fill_discard(q, group_id, 0);
        }

        inline AccessedDynamicIdGenerator<int_t, group_size> get_access(sycl::handler &cgh) {
            return {cgh, *this};
        }
    };

} // namespace shamalgs::atomic
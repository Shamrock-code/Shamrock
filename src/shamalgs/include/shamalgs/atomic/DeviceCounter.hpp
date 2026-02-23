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
 * @file DeviceCounter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::atomic {

    /**
     * @brief Utility to count group id on device
     *
     * @tparam int_t
     */
    template<class int_t>
    class DeviceCounter;

    template<class int_t>
    class AccessedDeviceCounter {
        public:
        sycl::accessor<int_t, 1, sycl::access::mode::read_write, sycl::access::target::device>
            counter;

        inline AccessedDeviceCounter(sycl::handler &cgh, DeviceCounter<int_t> &gen)
            : counter{gen.counter, cgh, sycl::read_write} {}

        template<sycl::memory_order order>
        inline sycl::atomic_ref<
            int_t,
            order,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>
        attach_atomic() const {
            return sycl::atomic_ref<
                int_t,
                order,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>(counter[0]);
        }
    };

    template<class int_t>
    class DeviceCounter {
        public:
        sycl::buffer<int_t> counter;

        inline explicit DeviceCounter(sycl::queue &q) : counter(1) {
            memory::buf_fill_discard(q, counter, int_t(0));
        }

        inline AccessedDeviceCounter<int_t> get_access(sycl::handler &cgh) { return {cgh, *this}; }
    };

} // namespace shamalgs::atomic

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file USMPtrHolder.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/USMPtrHolder.hpp"
#include "shambase/string.hpp"
#include "shambackends/details/internal_alloc.hpp"
#include <memory>

namespace sham {

    template<USMKindTarget target>
    USMPtrHolder<target> USMPtrHolder<target>::create(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment) {

        sycl::context &sycl_ctx = dev_sched->ctx->ctx;
        sycl::device &dev       = dev_sched->ctx->device->dev;
        void *usm_ptr           = details::internal_alloc<target>(sz, dev_sched, alignment);

        return USMPtrHolder<target>(usm_ptr, sz, dev_sched);
    }

    template<USMKindTarget target>
    USMPtrHolder<target>
    USMPtrHolder<target>::create_nullptr(std::shared_ptr<DeviceScheduler> dev_sched) {

        sycl::context &sycl_ctx = dev_sched->ctx->ctx;
        sycl::device &dev       = dev_sched->ctx->device->dev;
        void *usm_ptr           = nullptr;

        return USMPtrHolder<target>(usm_ptr, 0, dev_sched);
    }

    template<USMKindTarget target>
    void USMPtrHolder<target>::free_ptr() {
        if (usm_ptr != nullptr) {
            details::internal_free<target>(usm_ptr, get_bytesize(), dev_sched);
            usm_ptr = nullptr;
        }
    }

    template<USMKindTarget target>
    USMPtrHolder<target>::~USMPtrHolder() {
        free_ptr();
    }

    template class USMPtrHolder<device>;
    template class USMPtrHolder<shared>;
    template class USMPtrHolder<host>;

} // namespace sham

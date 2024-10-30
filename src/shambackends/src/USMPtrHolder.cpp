// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file USMPtrHolder.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/USMPtrHolder.hpp"
#include "shambase/string.hpp"
#include <memory>

namespace sham {

    template<USMKindTarget target>
    USMPtrHolder<target> USMPtrHolder<target>::create(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment) {

        sycl::context &sycl_ctx = dev_sched->ctx->ctx;
        sycl::device &dev       = dev_sched->ctx->device->dev;
        void *usm_ptr           = nullptr;

        if (sz > 0) {

            if (alignment) {

                // TODO upgrade alignment to 256-bit for CUDA ?

                if constexpr (target == device) {
                    usm_ptr = sycl::aligned_alloc_device(*alignment, sz, dev, sycl_ctx);
                } else if constexpr (target == shared) {
                    usm_ptr = sycl::aligned_alloc_shared(*alignment, sz, dev, sycl_ctx);
                } else if constexpr (target == host) {
                    usm_ptr = sycl::aligned_alloc_host(*alignment, sz, sycl_ctx);
                } else {
                    shambase::throw_unimplemented();
                }
            } else {
                if constexpr (target == device) {
                    usm_ptr = sycl::malloc_device(sz, dev, sycl_ctx);
                } else if constexpr (target == shared) {
                    usm_ptr = sycl::malloc_shared(sz, dev, sycl_ctx);
                } else if constexpr (target == host) {
                    usm_ptr = sycl::malloc_host(sz, sycl_ctx);
                } else {
                    shambase::throw_unimplemented();
                }
            }

            if (usm_ptr == nullptr) {
                std::string err_msg = "";
                if (alignment) {
                    err_msg = shambase::format(
                        "USM allocation failed, details : sz={}, target={}, alignment={}",
                        sz,
                        target,
                        *alignment);
                } else {
                    err_msg = shambase::format(
                        "USM allocation failed, details : sz={}, target={}", sz, target);
                }
                shambase::throw_with_loc<std::runtime_error>(err_msg);
            }
        }

        return USMPtrHolder<target>(usm_ptr, sz, dev_sched);
    }

    template<USMKindTarget target>
    void USMPtrHolder<target>::free_ptr() {
        if (usm_ptr != nullptr) {
            sycl::context &sycl_ctx = dev_sched->ctx->ctx;
            sycl::free(usm_ptr, sycl_ctx);
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

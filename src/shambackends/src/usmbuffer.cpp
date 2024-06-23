// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file usmbuffer.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/usmbuffer.hpp"
#include <memory>

namespace sham {

    template<USMKindTarget target>
    usmptr_holder<target>::usmptr_holder(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched) : dev_sched(dev_sched), size(sz) {

        sycl::context & sycl_ctx = dev_sched->ctx->ctx;
        sycl::device & dev = dev_sched->ctx->device->dev;

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

    template<USMKindTarget target>
    usmptr_holder<target>::~usmptr_holder() {
        sycl::context & sycl_ctx = dev_sched->ctx->ctx;
        sycl::free(usm_ptr, sycl_ctx);
    }

    template class usmptr_holder<device>;
    template class usmptr_holder<shared>;
    template class usmptr_holder<host>;

} // namespace sham
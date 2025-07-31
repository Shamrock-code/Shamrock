// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceContext.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceContext.hpp"
#include "shamcomm/logs.hpp"

namespace sham {

    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const &e) {
                printf("Caught synchronous SYCL exception: %s\n", e.what());
            }
        }
    };

    /**
     * @brief Lambda used to provide sycl::context initialization
     */
    auto ctx_init = [](std::shared_ptr<Device> &dev) -> sycl::context {
        if (!bool(dev)) {
            shambase::throw_with_loc<std::invalid_argument>("dev is empty");
        }
        return sycl::context(dev->dev, exception_handler);
    };

    DeviceContext::DeviceContext(std::shared_ptr<Device> dev)
        : device(std::move(dev)), ctx(ctx_init(device)) {}

    void DeviceContext::print_info() {
        device->print_info();

        shamcomm::logs::raw_ln("  Context info");

        // deprecated in DPCPP and does not really make sense in acpp either
        // logger::raw_ln("   - is_host() :", ctx.is_host());

        // #ifdef SYCL_COMP_ACPP
        //         logger::raw_ln("   - hipSYCL_hash_code() :", ctx.hipSYCL_hash_code());
        // #endif
    }

    bool DeviceContext::use_direct_comm() {

        if (!bool(device)) {
            shambase::throw_with_loc<std::runtime_error>("Why is device not allocated");
        }

        auto &dev = *device;
        if (dev.mpi_prop.is_mpi_direct_capable) {
            return true;
        } else {
            return false;
        }
    }

} // namespace sham

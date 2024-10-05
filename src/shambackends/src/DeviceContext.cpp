// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceContext.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

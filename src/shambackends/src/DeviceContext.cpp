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

    DeviceContext::DeviceContext(std::shared_ptr<Device> dev) : device(std::move(dev)) {

        if(bool(device)){
            ctx = sycl::context(device->dev,exception_handler);
        }else{
            shambase::throw_with_loc<std::invalid_argument>("dev is empty");
        }

    }

    void DeviceContext::print_info(){
        device->print_info();

        shamcomm::logs::raw_ln("  Context info");

        logger::raw_ln("   - is_host() :",ctx.is_host());
        #ifdef SYCL_COMP_ACPP
        logger::raw_ln("   - hipSYCL_hash_code() :",ctx.hipSYCL_hash_code());
        #endif
    }
    
} // namespace sham
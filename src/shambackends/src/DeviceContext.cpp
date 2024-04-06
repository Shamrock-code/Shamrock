#include "shambackends/DeviceContext.hpp"

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
    
} // namespace sham
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

    DeviceContext::DeviceContext(Device *dev) : device(dev), ctx(dev->dev, exception_handler) {}
    
} // namespace sham
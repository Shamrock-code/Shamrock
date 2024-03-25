#include "shambackends/Device.hpp"

namespace sham {

    class DeviceContext{
        public:

        Device * device;

        sycl::context ctx;

        DeviceContext(Device * device);

    };

}
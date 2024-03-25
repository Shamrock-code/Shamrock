#include "shambackends/DeviceQueue.hpp"

namespace sham {

    DeviceQueue::DeviceQueue(std::string queue_name, DeviceContext * ctx, bool in_order) :
        queue_name(queue_name), ctx(ctx) ,in_order(in_order)
    {
        if(in_order){
            q = sycl::queue{ctx->ctx, ctx->device->dev, sycl::property::queue::in_order{}};
        }else{
            q = sycl::queue{ctx->ctx, ctx->device->dev};
        }
        
    }
    
} // namespace sham
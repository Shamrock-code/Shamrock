#include "shambackends/DeviceScheduler.hpp"
#include "shambase/exception.hpp"

namespace sham {

    DeviceScheduler::DeviceScheduler(DeviceContext *ctx) : ctx(ctx) {

        queues.push_back(std::make_unique<DeviceQueue>("main_queue", ctx, false));
    }

    DeviceQueue &DeviceScheduler::get_queue(u32 i) { return *(queues.at(i)); }

} // namespace sham
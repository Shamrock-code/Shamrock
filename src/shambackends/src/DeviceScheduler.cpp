#include "shambackends/DeviceScheduler.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"

namespace sham {

    DeviceScheduler::DeviceScheduler(DeviceContext *ctx) : ctx(ctx) {

        queues.push_back(std::make_unique<DeviceQueue>("main_queue", ctx, false));
    }

    DeviceQueue &DeviceScheduler::get_queue(u32 i) { return *(queues.at(i)); }

    void DeviceScheduler::print_info(){
        for (auto & q : queues) {
            std::string tmp = shambase::format("name : {:20s} {}",q->queue_name, q->in_order);
            shamcomm::logs::raw_ln(tmp);
        }
    }

} // namespace sham
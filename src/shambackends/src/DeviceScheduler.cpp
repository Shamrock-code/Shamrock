// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceScheduler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceScheduler.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"

namespace sham {

    DeviceScheduler::DeviceScheduler(std::shared_ptr<DeviceContext> ctx) : ctx(ctx) {

        queues.push_back(std::make_unique<DeviceQueue>("main_queue", ctx, false));
    }

    DeviceQueue &DeviceScheduler::get_queue(u32 i) { return *(queues.at(i)); }

    void DeviceScheduler::print_info(){

        ctx->print_info();

        shamcomm::logs::raw_ln("  Queue list:");
        for (auto & q : queues) {
            std::string tmp = shambase::format("   - name : {:20s} in order : {}",q->queue_name, q->in_order);
            shamcomm::logs::raw_ln(tmp);
        }
    }

    void DeviceScheduler::test(){
        for (auto & q : queues) {
            q->test();
        }
    }

} // namespace sham
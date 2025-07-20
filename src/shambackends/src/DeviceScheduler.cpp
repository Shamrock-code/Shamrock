// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceScheduler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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

    void DeviceScheduler::print_info() {

        ctx->print_info();

        shamcomm::logs::raw_ln("  Queue list:");
        for (auto &q : queues) {
            std::string tmp
                = shambase::format("   - name : {:20s} in order : {}", q->queue_name, q->in_order);
            shamcomm::logs::raw_ln(tmp);
        }
    }

    void DeviceScheduler::test() {
        for (auto &q : queues) {
            q->test();
        }
    }

    bool DeviceScheduler::use_direct_comm() { return ctx->use_direct_comm(); }

} // namespace sham

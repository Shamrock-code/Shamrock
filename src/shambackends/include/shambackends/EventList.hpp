// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EventList.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/DeviceContext.hpp"

namespace sham {

    namespace details {
        class BufferEventHandler;
    }

    class EventList {
        public:
        inline void apply_dependancy(sycl::handler &h) { h.depends_on(events); }

        inline void wait() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait();
            }
            consumed = true;
        }

        inline void wait_and_throw() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait_and_throw();
            }
            consumed = true;
        }

        inline void add_event(sycl::event e) {
            events.push_back(e);
            consumed = false;
        }

        inline std::string get_state() {
            return shambase::format("events : {}, consumed : {}", events.size(), consumed);
        }

        EventList(SourceLocation loc = SourceLocation{}) : loc_build(loc) {}

        ~EventList();

        private:
        std::vector<sycl::event> events = {};
        bool consumed                   = false;
        SourceLocation loc_build;

        friend class DeviceQueue;
        friend class details::BufferEventHandler;
    };
} // namespace sham
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BufferEventHandler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"
#include <shambackends/details/BufferEventHandler.hpp>
#include <stdexcept>

namespace sham::details {

    void BufferEventHandler::filter_events() {

        auto is_event_complete = [](const sycl::event &event) -> bool {
            return (
                event.get_info<sycl::info::event::command_execution_status>()
                == sycl::info::event_command_status::complete);
        };

        auto filter_events = [&](std::vector<sycl::event> &events) -> std::vector<sycl::event> {
            std::vector<sycl::event> ret;
            for (auto &e : events) {
                if (!is_event_complete(e)) {
                    ret.emplace_back(std::move(e));
                } else {
                    // discard the event
                }
            }

            return ret;
        };

        read_events  = filter_events(read_events);
        write_events = filter_events(write_events);
    }

    void BufferEventHandler::read_access(sham::EventList &depends_list, SourceLocation src_loc) {

        if (!up_to_date_events) {
            std::string err_msg = shambase::format(
                "you have requested a read access on a buffer in an incomplete state\n"
                "  read_access call location : {}\n"
                "  last access location : {}\n",
                src_loc.format_one_line(),
                last_access_loc.format_one_line());

            shambase::throw_with_loc<std::runtime_error>(err_msg);
        }

        filter_events();

        up_to_date_events = false;
        last_access       = READ;

        for (sycl::event e : write_events) {
            depends_list.add_event(e);
        }

        last_access_loc = src_loc;
    }

    void BufferEventHandler::write_access(sham::EventList &depends_list, SourceLocation src_loc) {

        if (!up_to_date_events) {
            std::string err_msg = shambase::format(
                "you have requested a write access on a buffer in an incomplete state\n"
                "  write_access call location : {}\n"
                "  last access location : {}\n",
                src_loc.format_one_line(),
                last_access_loc.format_one_line());

            shambase::throw_with_loc<std::runtime_error>(err_msg);
        }

        filter_events();

        up_to_date_events = false;
        last_access       = WRITE;

        for (sycl::event e : write_events) {
            depends_list.add_event(e);
        }
        for (sycl::event e : read_events) {
            depends_list.add_event(e);
        }

        last_access_loc = src_loc;
    }

    void BufferEventHandler::complete_state(sycl::event e, SourceLocation src_loc) {
        complete_state(std::vector<sycl::event>{e}, src_loc);
    }

    void BufferEventHandler::complete_state(
        const std::vector<sycl::event> &events, SourceLocation src_loc) {

        StackEntry stack_loc{};

        if (up_to_date_events) {
            shambase::throw_with_loc<std::runtime_error>(
                "the event state of that buffer is already complete"
                "complete_state call location : "
                + src_loc.format_one_line());
        }

        filter_events();

        if (last_access == READ) {

            for (auto e : events) {
                read_events.push_back(e);
            }

            up_to_date_events = true;

        } else if (last_access == WRITE) {

            // the new event depends on those so we can clear
            write_events.clear();
            read_events.clear();

            for (auto e : events) {
                write_events.push_back(e);
            }

            up_to_date_events = true;
        }
    }

} // namespace sham::details

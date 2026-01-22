// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BufferEventHandler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the class definition for BufferEventHandler.
 */

#include "shambase/SourceLocation.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl.hpp"
#include <shambackends/details/BufferEventHandler.hpp>

namespace sham::details {

    /**
     * @brief Class that manages the event state of a SYCL USM buffer
     *
     * This class is used to manage the accesses to a SYCL USM buffer. It keeps track of the events
     * related to read and write operations of the buffer. The class has two states: incomplete
     * and complete.
     *
     * - When the class is in an incomplete state, this is the case when an access was queried (by
     * calling `read_access` or `write_access`) until the event is registered using
     * `complete_state`.
     * - When the class is in a complete state, it means that the buffer access events are up to
     * date.
     *
     * The class is designed to work with SYCL USM buffers.
     */
    class BufferEventHandler {

        public:
        /**
         * @brief Vector of events related to read operations on the buffer
         *
         * This vector keeps track of the events that correcpond to read operation on the managed
         * object
         */
        std::vector<sycl::event> read_events;

        /**
         * @brief Vector of events related to write operations on the buffer
         *
         * This vector keeps track of the events that correcpond to wire operation on the managed
         * object
         */
        std::vector<sycl::event> write_events;

        /**
         * @brief Wait for all the buffer accesses to be completed
         *
         * This function waits for all the buffer accesses to be completed. It
         * waits for both read and write events to be completed.
         *
         * @param src_loc Source location of the call to this function
         */
        inline void wait_all(SourceLocation src_loc = SourceLocation{}) {
            sycl::event::wait(read_events);
            sycl::event::wait(write_events);

            read_events.clear();
            write_events.clear();
        }

        /**
         * @brief Check if there if the events lists are empty
         *
         * @return `true` if both buffer events lists are empty, `false` otherwise
         */
        inline bool is_empty() { return read_events.empty() && write_events.empty(); }

        /**
         * @brief Enum to represent the last operation performed on the buffer
         *
         * This enum is used to represent the last operation on the managed object.
         * It can be either a read operation or a write operation.
         */
        enum last_op { READ, WRITE } last_access;

        /**
         * @brief Flag to indicate if the buffer access events are up to date
         *
         * If it is `true`, it means that the buffer access events are up to date.
         * If it is `false`, it means that the buffer access events are not up to date, the event
         * handler is in a incomplete state.
         */
        bool up_to_date_events = true;

        /**
         * @brief Source location of the last access to the buffer
         *
         * This is the source location of the last access to the buffer. It is used to
         * report an error if the buffer is accessed in an incomplete state.
         */
        SourceLocation last_access_loc;

        /**
         * @brief Adds events conditioning the validity of a buffer for read access to the
         * dependency list. Also sets the event handler to incomplete state (`up_to_date_events` =
         * false).
         *
         * @details This function is used to add events that condition the validity of a buffer
         *          for read access to the dependency list. It is typically used when a read
         *          operation is about to be performed on the buffer. The function adds the
         *          events related to previous write operations on the buffer to the dependency
         *          list.
         *
         * @param depends_list The vector of events that the read access depends on.
         * @param src_loc The source location of the call to this function.
         *
         * @throws std::runtime_error if the buffer event handler is in an incomplete state.
         *         This exception is thrown when the function is called on a buffer event
         *         handler that is in an incomplete state (i.e., `up_to_date_events` is `false`).
         */
        void read_access(sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{});

        /**
         * @brief Adds events conditioning the validity of a buffer for write access to the
         * dependency list. Also sets the event handler to incomplete state (`up_to_date_events` =
         * false).
         *
         * @details This function is used to add events that condition the validity of a buffer
         *          for write access to the dependency list. It is typically used when a write
         *          operation is about to be performed on the buffer. The function adds the
         *          events related to previous read and write operations on the buffer to the
         *          dependency list.
         *
         * @param depends_list The vector of events that the write access depends on.
         * @param src_loc The source location of the call to this function.
         *
         * @throws std::runtime_error if the buffer event handler is in an incomplete state.
         *         This exception is thrown when the function is called on a buffer event
         *         handler that is in an incomplete state (i.e., `up_to_date_events` is `false`).
         */
        void write_access(sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{});

        /**
         * @brief Completes the state of the buffer event handler with the event for which the
         * access was queried.
         *
         * This function is used to complete the state of the buffer event handler with the event
         * for which the access was queried. Once the state is completed, the buffer event handler
         * is marked as up to date.
         *
         * @param e The event for which the access was queried.
         * @param src_loc The source location of the call to this function.
         *
         * @throws std::runtime_error if the buffer event handler is already up to date.
         *         This exception is thrown when the function is called on a buffer event handler
         * that is already up to date (i.e., `up_to_date_events` is `true`).
         */
        void complete_state(sycl::event e, SourceLocation src_loc = SourceLocation{});

        /**
         * @brief Completes the state of the buffer event handler with the specified events.
         *
         * This function is used to complete the state of the buffer event handler with the given
         * list of events. Once the state is completed, the buffer event handler is marked as up to
         * date.
         *
         * @param events A vector of events for which the state is completed.
         * @param src_loc The source location of the call to this function.
         *
         * @throws std::runtime_error if the buffer event handler is already up to date.
         *         This exception is thrown when the function is called on a buffer event handler
         * that is already up to date (i.e., `up_to_date_events` is `true`).
         */
        void complete_state(
            const std::vector<sycl::event> &events, SourceLocation src_loc = SourceLocation{});

        /**
         * @brief Filter the read and write events so that only pending events are stored.
         *
         * Events that are completed are removed from the read and write event lists.
         */
        void filter_events();

        /**
         * @brief Completes the state of the buffer event handler using an EventList.
         *
         * This function completes the state of the buffer event handler with the events
         * contained in the provided EventList. Once the state is completed, the events
         * in the EventList are marked as consumed.
         *
         * @param events The EventList containing events for which the state is completed.
         */
        inline void complete_state(sham::EventList &events) {
            complete_state(events.events);
            events.consumed = true;
        }
    };

} // namespace sham::details

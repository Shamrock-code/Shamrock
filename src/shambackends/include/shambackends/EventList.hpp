// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EventList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/DeviceContext.hpp"

namespace sham {

    namespace details {
        // forward declare BufferEventHandler
        class BufferEventHandler;
    } // namespace details

    /**
     * @brief Class to manage a list of SYCL events.
     */
    class EventList {
        public:
        /**
         * @brief Apply all events in the list as dependancies to a SYCL command group.
         * @param h The SYCL command group handler.
         *
         * Example of usage:
         * @code {.cpp}
         * EventList events;
         * <... add events ...>
         * sycl::queue q{};
         * q.submit([&](sycl::handler &h) {
         *     events.apply_dependancy(h);
         *     // ...
         * });
         *
         * @endcode
         */
        inline void apply_dependancy(sycl::handler &h) { h.depends_on(events); }

        /**
         * @brief Wait for all events in the list to be finished.
         *
         * Waits for all events in the list to be finished. After calling this function, the
         * list is considered consumed.
         */
        inline void wait() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait();
            }
            consumed = true;
        }

        /**
         * @brief Wait for all events in the list to be finished and throw an exception if one has
         * occurred.
         *
         * Waits for all events in the list to be finished. And throws an exception if one has
         * occurred in any of the events.
         */
        inline void wait_and_throw() {
            StackEntry __s{};
            for (auto &e : events) {
                e.wait_and_throw();
            }
            consumed = true;
        }

        /**
         * @brief Add an event to the list of events.
         *
         * Add a SYCL event to the list of events.
         *
         * @param e The SYCL event to add.
         */
        inline void add_event(sycl::event e) {
            events.push_back(e);
            consumed = false;
        }

        /**
         * @brief Add multiple events to the list of events.
         *
         * Inserts a list of SYCL events into the event list. The list is marked
         * as not consumed after the insertion.
         *
         * @param e A reference to a vector containing the SYCL events to add.
         */
        inline void add_events(std::vector<sycl::event> &e) {
            events.insert(events.end(), e.begin(), e.end());
            consumed = false;
        }

        /**
         * @brief Add all events from another EventList to this one.
         *
         * Insert all events from another EventList into this one. The list is marked
         * as not consumed after the insertion. The other list is marked as consumed
         * after the insertion.
         *
         * @param e The EventList from which to add all events.
         */
        inline void add_events(sham::EventList &e) {
            events.insert(events.end(), e.events.begin(), e.events.end());
            consumed   = false;
            e.consumed = true;
        }

        /**
         * @brief Get a string representation of the EventList's state.
         *
         * This function returns a string representation of the EventList's state,
         * including the number of events in the list and whether the list is
         * considered consumed.
         *
         * @return A string representation of the EventList's state.
         */
        inline std::string get_state() {
            return shambase::format("events : {}, consumed : {}", events.size(), consumed);
        }

        /// Default constructor for EventList with source location
        EventList(SourceLocation loc = SourceLocation{}) : loc_build(loc) {}

        /// same constructor but with initializer list
        EventList(std::initializer_list<sycl::event> e, SourceLocation loc = SourceLocation{})
            : events(e), loc_build(loc) {}

        /**
         * @brief Destructor for EventList.
         *
         * This destructor checks if the EventList has not been consumed and
         * still contains events. If so, it logs an error message and throws
         * a runtime exception after waiting for all events.
         */
        ~EventList();

        /// Get the list of events
        inline std::vector<sycl::event> &get_events() { return events; }

        /// Get the list of events (const)
        inline const std::vector<sycl::event> &get_events() const { return events; }

        /// Set the consumed state of the EventList (to be used with interop)
        inline void set_consumed(bool consumed) { this->consumed = consumed; }

        private:
        std::vector<sycl::event> events = {};    ///< The list of SYCL events
        bool consumed                   = false; ///< Whether the list is considered consumed
        SourceLocation loc_build;                ///< The source location of the EventList creation

        friend class DeviceQueue;
        friend class details::BufferEventHandler;
    };
} // namespace sham

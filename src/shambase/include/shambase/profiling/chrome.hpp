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
 * @file chrome.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <string>

namespace shambase::profiling::chrome {

    /**
     * @brief Set the Chrome tracing process id
     *
     * This sets the process id used by the Chrome tracing library.
     * It should be set to the value of the MPI world rank.
     *
     * @param pid The process id
     */
    void set_chrome_pid(u32 pid);

    /**
     * @brief Set the time offset used for Chrome tracing
     *
     * This sets the time offset used for Chrome tracing.
     * It is used to adjust the times of the events so that they are correctly
     * aligned with the times of the other events in the trace.
     *
     * @param offset The time offset
     */
    void set_time_offset(f64 offset);

    /**
     * @brief Register the start of an event in Chrome tracing
     *
     * This registers the start of an event in Chrome tracing.
     * It is used to mark the beginning of an event in the trace.
     *
     * @param name The name of the event
     * @param category_name The category name of the event
     * @param t_start The start time of the event
     * @param pid The process id of the event
     * @param tid The thread id of the event
     */
    void register_event_start(
        const std::string &name, const std::string &category_name, f64 t_start, u64 pid, u64 tid);

    /**
     * @brief Register the end of an event in Chrome tracing
     *
     * This registers the end of an event in Chrome tracing.
     * It is used to mark the end of an event in the trace.
     *
     * @param name The name of the event
     * @param category_name The category name of the event
     * @param tend The end time of the event
     * @param pid The process id of the event
     * @param tid The thread id of the event
     */
    void register_event_end(
        const std::string &name, const std::string &category_name, f64 tend, u64 pid, u64 tid);

    /**
     * @brief Register a complete event in Chrome tracing
     *
     * This registers a complete event in Chrome tracing.
     * It is used to register an event with both its start and end times.
     *
     * @param name The name of the event
     * @param category_name The category name of the event
     * @param t_start The start time of the event
     * @param tend The end time of the event
     * @param pid The process id of the event
     * @param tid The thread id of the event
     */
    void register_event_complete(
        const std::string &name,
        const std::string &category_name,
        f64 t_start,
        f64 tend,
        u64 pid,
        u64 tid);

    /**
     * @brief Register a thread name in Chrome tracing
     *
     * This registers a thread name in Chrome tracing.
     * It is used to give a name to a thread in the trace.
     *
     * @param pid The process id of the thread
     * @param tid The thread id of the thread
     * @param name The name of the thread
     */
    void register_metadata_thread_name(u64 pid, u64 tid, const std::string &name);

    /**
     * @brief Register a counter value in Chrome tracing
     *
     * This registers a counter value in Chrome tracing.
     * It is used to record a value of a counter in the trace.
     *
     * @param pid The process id of the counter
     * @param t The time of the counter
     * @param name The name of the counter
     * @param val The value of the counter
     */
    void register_counter_val(u64 pid, f64 t, const std::string &name, f64 val);

} // namespace shambase::profiling::chrome

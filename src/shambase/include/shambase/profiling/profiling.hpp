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
 * @file profiling.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <optional>
#include <string>

namespace shambase::profiling {

    /// @brief Check if profiling is enabled
    bool is_profiling_enabled();

    /**
     * @brief Set wether to enable NVTX profiling
     *
     * @param enable_nvtx Whether to enable NVTX profiling
     */
    void set_enable_nvtx(bool enable_nvtx);

    /**
     * @brief Set wether to enable profiling
     *
     * @param enable_profiling Whether to enable profiling
     */
    void set_enable_profiling(bool enable_profiling);

    /**
     * @brief Use complete event, or start and begin event in chrome tracing
     *
     * @param use_complete_event Whether to use complete events
     */
    void set_use_complete_event(bool use_complete_event);

    /**
     * @brief Set the event record threshold
     *
     * The threshold is the minimum time in seconds between two events, if the time between two
     * events is smaller than the threshold, the second event will not be recorded.
     *
     * @param threshold The event record threshold
     */
    void set_event_record_threshold(f64 threshold);

    /**
     * @brief Register the start of a profiling event
     *
     * @param fileloc The source location of the function
     * @param t_start The starting time of the event
     * @param name The name of the event
     * @param category_name The name of the category
     */
    void stack_entry_start(
        const SourceLocation &fileloc,
        f64 t_start,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    /**
     * @brief Register the end of a profiling event
     *
     * @param fileloc The source location of the function
     * @param t_start The starting time of the event
     * @param tend The ending time of the event
     * @param name The name of the event
     * @param category_name The name of the category
     */
    void stack_entry_end(
        const SourceLocation &fileloc,
        f64 t_start,
        f64 tend,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    /**
     * @brief Start a profiling event without a time info
     *
     * @param fileloc The source location of the function
     * @param name The name of the event
     * @param category_name The name of the category
     */
    void stack_entry_start_no_time(
        const SourceLocation &fileloc,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    /**
     * @brief End a profiling event without a time info
     *
     * @param fileloc The source location of the function
     * @param name The name of the event
     * @param category_name The name of the category
     */
    void stack_entry_end_no_time(
        const SourceLocation &fileloc,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    /**
     * @brief Register a counter value
     *
     * @param name The name of the counter
     * @param time The time of the event
     * @param val The value of the counter
     */
    void register_counter_val(const std::string &name, f64 time, f64 val);

} // namespace shambase::profiling

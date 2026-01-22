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
 * @file stacktrace.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the definition for the stacktrace related functionality.
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include "shambase/unique_name_macro.hpp"
#include <stack>

namespace shambase::details {

    /**
     * @brief Returns the current wall clock time in seconds.
     *
     * @return f64 The current wall clock time in seconds.
     */
    f64 get_wtime();

#ifdef SHAMROCK_USE_PROFILING

    /**
     * @brief Register the start of a profile entry.
     * This is required for chrome profiling as there is a separate entry for start and end
     *
     * @param loc The source location of the profile entry.
     * @param start_time The start time of the profile entry.
     */
    void register_profile_entry_start(std::source_location loc, f64 start_time);

    /**
     * @brief Register a profile entry.
     * This register the end of a profile entry for chrome tracing and a complete entry in the
     * builtin profiling data
     *
     * @param loc The source location of the profile entry.
     * @param start_time The start time of the profile entry.
     * @param end_time The end time of the profile entry.
     */
    void register_profile_entry(std::source_location loc, f64 start_time, f64 end_time);

    /**
     * @brief Dump the profiling data in a JSON format to a file.
     *
     * @param process_prefix The prefix of the process name.
     * @param world_rank The rank of the process.
     */
    void dump_profilings(std::string process_prefix, u32 world_rank);

    /**
     * @brief Dump the profiling data in a Chrome Tracing format.
     *
     * @param process_prefix The prefix of the process name.
     * @param world_rank The rank of the process.
     */
    void dump_profilings_chrome(std::string process_prefix, u32 world_rank);

    /**
     * @brief Clear the profiling data. (should be done in large run to avoid out-of-memory)
     */
    void clear_profiling_data();

#endif

    /**
     * @brief The call stack used to keep track of the stack trace.
     * It is used to print the stack trace when an exception is thrown.
     */
    inline std::stack<SourceLocation> call_stack;

    /// Helper class to manage the call stack entry
    struct CallStackEntry {
        inline CallStackEntry(SourceLocation &loc) {
            // Push the source location to the call stack
            call_stack.emplace(loc);
        }

        // This class is not safe if copied or moved

        CallStackEntry(const CallStackEntry &)            = delete;
        CallStackEntry &operator=(const CallStackEntry &) = delete;
        CallStackEntry(CallStackEntry &&)                 = delete;
        CallStackEntry &operator=(CallStackEntry &&)      = delete;

        inline ~CallStackEntry() {
            // Pop the source location from the call stack
            call_stack.pop();
        }
    };

    struct BasicStackEntry {
        SourceLocation loc; ///< Source location attached to the entry
        bool do_timer;      ///< is the timer enabled for this entry

#ifdef SHAMROCK_USE_PROFILING
        f64 wtime_start; ///< start time of the entry
#endif

        CallStackEntry scoped_callstack_entry; ///< scoped call stack entry

        /**
         * @brief Construct a new Basic Stack Entry object.
         *
         * @param do_timer Is the timer enabled for this entry (default: true)
         * @param loc Source location attached to the entry (default: SourceLocation{})
         */
        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer), scoped_callstack_entry(loc) {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                wtime_start = get_wtime();
                shambase::profiling::stack_entry_start(loc, wtime_start);
            } else {
                shambase::profiling::stack_entry_start_no_time(loc);
            }
#endif
        }

        /**
         * @brief Construct a new Basic Stack Entry object.
         *
         * @param do_timer Is the timer enabled for this entry (default: true)
         * @param loc Source location attached to the entry (default: SourceLocation{})
         */
        inline BasicStackEntry(
            SourceLocation &callsite, bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer), scoped_callstack_entry(callsite) {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                wtime_start = get_wtime();
                shambase::profiling::stack_entry_start(loc, wtime_start);
            } else {
                shambase::profiling::stack_entry_start_no_time(loc);
            }
#endif
        }

        /**
         * @brief Destroy the Basic Stack Entry object.
         *
         * Pop the source location from the call stack and stop the timer if enabled.
         */
        inline ~BasicStackEntry() {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                f64 wtime_end = get_wtime();
                shambase::profiling::stack_entry_end(loc, wtime_start, wtime_end);
            } else {
                shambase::profiling::stack_entry_end_no_time(loc);
            }
#endif
        }
    };

    struct NamedBasicStackEntry {
        SourceLocation loc; ///< Source location attached to the entry
        bool do_timer;      ///< is the timer enabled for this entry
        std::string name;   ///< Name of the entry

#ifdef SHAMROCK_USE_PROFILING
        f64 wtime_start; ///< start time of the entry
#endif

        CallStackEntry scoped_callstack_entry; ///< scoped call stack entry

        /**
         * @brief Construct a new Named Basic Stack Entry object
         *
         * @param name Name of the entry
         * @param do_timer Is the timer enabled for this entry (default: true)
         * @param loc Source location attached to the entry (default: SourceLocation{})
         */
        inline NamedBasicStackEntry(
            std::string name, bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : name(name), loc(loc), do_timer(do_timer), scoped_callstack_entry(loc) {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                wtime_start = get_wtime();
                shambase::profiling::stack_entry_start(loc, wtime_start, name);
            } else {
                shambase::profiling::stack_entry_start_no_time(loc, name);
            }
#endif
        }

        /**
         * @brief Destroy the Named Basic Stack Entry object
         *
         * Pop the name from the call stack and stop the timer if enabled
         */
        inline ~NamedBasicStackEntry() {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                f64 wtime_end = get_wtime();
                shambase::profiling::stack_entry_end(loc, wtime_start, wtime_end, name);
            } else {
                shambase::profiling::stack_entry_end_no_time(loc);
            }
#endif
        }
    };

} // namespace shambase::details

namespace shambase {

    /**
     * @brief Get the formatted callstack.
     *
     * This function returns a formatted string representing the current call stack.
     *
     * @return The formatted call stack as a string.
     */
    std::string fmt_callstack();

} // namespace shambase

/**
 * @brief Alias for shambase::details::BasicStackEntry.
 *
 * This alias is used to simplify the use of the BasicStackEntry class.
 */
using StackEntry = shambase::details::BasicStackEntry;

/**
 * @brief Alias for shambase::details::NamedBasicStackEntry.
 *
 * This alias is used to simplify the use of the NamedBasicStackEntry class.
 */
using NamedStackEntry = shambase::details::NamedBasicStackEntry;

/**
 * @brief Macro to create a stack entry.
 *
 * This macro defines a `StackEntry` variable with a unique name by leveraging
 * the `__shamrock_unique_name` macro.
 */
#define __shamrock_stack_entry()                                                                   \
    [[maybe_unused]] StackEntry __shamrock_unique_name(stack_loc_) {}

/**
 * @brief Macro to create a stack entry from a given location. Can be used only on SourceLocation &&
 *
 * This macro defines a `StackEntry` variable with a unique name by leveraging
 * the `__shamrock_unique_name` macro.
 */
#define __shamrock_log_callsite(callsite)                                                          \
    [[maybe_unused]] shambase::details::CallStackEntry __shamrock_unique_name(call_site_loc) {     \
        callsite                                                                                   \
    }

/**
 * @brief Macro to create a stack entry.
 *
 * This macro defines a `StackEntry` variable with a unique name by leveraging
 * the `__shamrock_unique_name` macro.
 */
#define __shamrock_stack_entry_with_callsite(callsite)                                             \
    [[maybe_unused]] StackEntry __shamrock_unique_name(stack_loc_) { callsite }

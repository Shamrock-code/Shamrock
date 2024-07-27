// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file stacktrace.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the definition for the stacktrace related functionality.
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/string.hpp"
#include "shambase/SourceLocation.hpp"
#include <stack>

#ifdef SHAMROCK_USE_NVTX
    #include <nvtx3/nvtx3.hpp>
#endif

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

    struct BasicStackEntry {
        SourceLocation loc; ///< Source location attached to the entry
        bool do_timer;      ///< is the timer enabled for this entry

#ifdef SHAMROCK_USE_PROFILING
        f64 wtime_start; ///< start time of the entry
#endif
        /**
         * @brief Construct a new Basic Stack Entry object.
         *
         * @param do_timer Is the timer enabled for this entry (default: true)
         * @param loc Source location attached to the entry (default: SourceLocation{})
         */
        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer) {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                // Start the timer if enabled
                wtime_start = get_wtime();
                register_profile_entry_start(loc.loc, wtime_start);
            }
#endif
            // Push the source location to the call stack
            call_stack.emplace(loc);
#ifdef SHAMROCK_USE_NVTX
            // Push a NVTX range
            nvtxRangePush(loc.loc.function_name());
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
                // Stop the timer if enabled
                f64 wtime_end = get_wtime();
                register_profile_entry(loc.loc, wtime_start, wtime_end);
            }
#endif
            // Pop the source location from the call stack
            call_stack.pop();
#ifdef SHAMROCK_USE_NVTX
            // Pop the NVTX range
            nvtxRangePop();
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

        /**
         * @brief Construct a new Named Basic Stack Entry object
         *
         * @param name Name of the entry
         * @param do_timer Is the timer enabled for this entry (default: true)
         * @param loc Source location attached to the entry (default: SourceLocation{})
         */
        inline NamedBasicStackEntry(
            std::string name, bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : name(name), loc(loc), do_timer(do_timer) {
#ifdef SHAMROCK_USE_PROFILING
            if (do_timer) {
                // Start the timer if enabled
                wtime_start = get_wtime();
            }
#endif
            call_stack.emplace(loc);
#ifdef SHAMROCK_USE_NVTX
            // Push a NVTX range
            nvtxRangePush(name.c_str());
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
                register_profile_entry(loc.loc, wtime_start, wtime_end);
            }
#endif
            call_stack.pop();
#ifdef SHAMROCK_USE_NVTX
            // Pop the NVTX range
            nvtxRangePop();
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

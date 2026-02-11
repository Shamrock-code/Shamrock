// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file stacktrace.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace shambase::details {

    /// Chrome tracing profile entry
    struct ChromeProfileEntry {
        std::string name; ///< Name of the profile entry
        u64 time_val;     ///< Time value for the profile entry
        bool is_start;    ///< Flag indicating if it is the start of the profile entry

        /**
         * @brief Format the Chrome profile entry
         * @param world_rank The MPI world rank for the profile entry
         * @return Formatted json string representing the profile entry
         */
        std::string format(u32 world_rank);
    };

    std::string ChromeProfileEntry::format(u32 world_rank) {
        if (is_start) {
            return shambase::format_printf(
                R"({
                "cat": "%s",
                "pid": %d,
                "tid": %d,
                "ts": %zu,
                "ph": "B",
                "name": "%s",
                "args": {
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());

        } else {
            return shambase::format_printf(
                R"({
                "cat": "%s",
                "pid": %d,
                "tid": %d,
                "ts": %zu,
                "ph": "E",
                "name": "%s",
                "args": {
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());
        }
    }

    /// Chrome tracing entries storage
    std::vector<ChromeProfileEntry> profile_data_chrome;

    /**
     * @brief Add a Chrome tracing entry to the storage
     *
     * @param loc The source location of the entry
     * @param time The time of the entry
     * @param is_start Whether the entry is a start or end entry
     */
    inline void add_entry_chrome(std::source_location loc, f64 time, bool is_start) {
        // Convert time to microseconds
        auto to_prof_time = [](f64 in) {
            return static_cast<u64>(in * 1e6);
        };
        // Add the entry to the storage
        profile_data_chrome.push_back(
            ChromeProfileEntry{loc.function_name(), to_prof_time(time), is_start});
    }

    /**
     * @brief Clear the Chrome tracing entries storage
     */
    inline void clear_chrome_entry() { profile_data_chrome.clear(); }

    void dump_profilings_chrome(std::string process_prefix, u32 world_rank) {

        // Open the file for writing
        std::ofstream outfile(process_prefix + std::to_string(world_rank));
        // Write the start of the JSON array
        outfile << "[";

        // Write each entry
        u32 len = profile_data_chrome.size();

        for (u32 i = 0; i < len; i++) {
            // Write the entry in the JSON format
            outfile << profile_data_chrome[i].format(world_rank);
            // Add a comma if it's not the last entry
            if (i != len - 1) {
                outfile << ",";
            }
        }

        // Write the end of the JSON array
        outfile << "]";
        // Close the file
        outfile.close();
    }

} // namespace shambase::details

namespace shambase::details {

    /// Utility to create a timer and start it
    auto make_timer = []() -> Timer {
        Timer tmp;
        tmp.start();
        return tmp;
    };

    /// Wall time global timer
    Timer global_timer = make_timer();

    // two entry types,
    //  one with start, end
    //  one with start, end as separate envents

    /**
     * @struct ProfileEntry
     * @brief Structure to hold data for a profiling entry
     *
     * This structure holds the start and end time of a profiling entry,
     * and the name of the entry.
     */
    struct ProfileEntry {
        f64 time_start;         ///< Start time of the profiling entry (in sec since programm start)
        f64 time_end;           ///< End time of the profiling entry (in sec since programm start)
        std::string entry_name; ///< Name of the profiling entry

        /**
         * @brief Format the profile entry as a JSON string
         *
         * @return std::string JSON string representation of the profile entry
         */
        std::string format() {
            return shambase::format_printf(
                R"({"tstart": %f, "tend": %f, "name": "%s"})", time_start, time_end, entry_name);
        }
    };

    /**
     * @brief Vector to hold profiling entries
     */
    std::vector<ProfileEntry> profile_data;

    /**
     * @brief Get the current wall time
     *
     * @return f64 Wall time in seconds since program start
     */
    f64 get_wtime() {
        global_timer.end();
        return global_timer.elasped_sec();
    }

    void register_profile_entry_start(std::source_location loc, f64 start_time) {
        add_entry_chrome(loc, start_time, true);
    };

    void register_profile_entry(std::source_location loc, f64 start_time, f64 end_time) {
        // Add the profile entry to the storage
        profile_data.push_back({start_time, end_time, loc.function_name()});
        // Add a Chrome profiling entry to the storage
        add_entry_chrome(loc, end_time, false);
    };

    void clear_profiling_data() {
        profile_data.clear();
        clear_chrome_entry();
    }

    void dump_profilings(std::string process_prefix, u32 world_rank) {
        std::ofstream outfile(process_prefix + std::to_string(world_rank));
        outfile << "[";

        u32 len = profile_data.size();

        for (u32 i = 0; i < len; i++) {
            outfile << profile_data[i].format();
            if (i != len - 1) {
                outfile << ",";
            }
        }

        outfile << "]";
        outfile.close();
    }

} // namespace shambase::details

namespace shambase {

    std::string _callstack_process_identifier;

    void set_callstack_process_identifier(std::string identifier) {
        _callstack_process_identifier = std::move(identifier);
    }

    std::vector<std::function<std::string()>> _callstack_gen_info_generators;

    void add_callstack_gen_info_generator(std::string (*generator)()) {
        _callstack_gen_info_generators.push_back(std::move(generator));
    }

    /**
     * @brief get the formatted callstack
     *
     * @return std::string
     */
    std::string fmt_callstack() {
        std::stack<SourceLocation> cpy = details::call_stack;

        std::vector<std::string> lines;

        while (!cpy.empty()) {
            SourceLocation l = cpy.top();
            lines.push_back(l.format_one_line_func());
            cpy.pop();
        }

        std::reverse(lines.begin(), lines.end());

        std::stringstream ss;

        if (!_callstack_process_identifier.empty()) {
            ss << "  -> process identifier: " << _callstack_process_identifier << "\n";
        }

        for (u32 i = 0; i < lines.size(); i++) {
            ss << shambase::format(" {:2} : {}\n", i, lines[i]);
        }

        for (auto &generator : _callstack_gen_info_generators) {
            ss << generator() << "\n";
        }

        return ss.str();
    }

} // namespace shambase

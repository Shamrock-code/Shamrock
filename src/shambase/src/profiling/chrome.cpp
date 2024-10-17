// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file chrome.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambase/profiling/chrome.hpp"
#include <fstream>

namespace {

    f64 to_prof_time(f64 in) { return in * 1e6; };

    void write(std::string event) {

        // static file stream storage
        static std::ofstream stream("shamrock_chrome_trace.json");

        static bool first = true;
        if (first) {
            stream << "{\"traceEvents\": [\n";
            first = false;
        }

        stream << event;
    }

    void register_event(std::string event) {

        write(event);
        return;

        // should write everyting on destructor call
        // like using a unique ptr to hold the buffer
        // push in the buffer only if the thing allocated to still dump profiling
        // if the buffer was destroyed but with other thing left to destroy
        static std::vector<std::string> event_buf;

        event_buf.push_back(event);

        if (event_buf.size() >= 1000) {
            for (auto &e : event_buf) {
                write(e);
            }
            event_buf.clear();
        }
    }

} // namespace

void shambase::profiling::register_event_start(
    const std::string &name, const std::string &display_name, f64 t_start, u64 pid, u64 tid) {
    register_event(shambase::format(
        "{{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"B\", \"ts\": {}, \"pid\": {}, \"tid\": "
        "{}}},\n",
        name,
        display_name,
        to_prof_time(t_start),
        pid,
        tid));
}

void shambase::profiling::register_event_end(
    const std::string &name, const std::string &display_name, f64 tend, u64 pid, u64 tid) {
    register_event(shambase::format(
        "{{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"E\", \"ts\": {}, \"pid\": {}, \"tid\": "
        "{}}},\n",
        name,
        display_name,
        to_prof_time(tend),
        pid,
        tid));
}

void shambase::profiling::register_metadata_thread_name(u64 pid, u64 tid, const std::string &name) {
    register_event("");
}

void shambase::profiling::register_counter_val(u64 pid, f64 t, const std::string &name, f64 val) {
    // { "pid": 0, "ph": "C", "ts":  0, "args": {"GPU mem usage":  0}},
    register_event(shambase::format(
        "{{ \"pid\": {}, \"ph\": \"C\", \"ts\":  {}, \"args\": {{\"{}\": {} }} }},",
        pid,
        to_prof_time(t),
        name,
        val));
}

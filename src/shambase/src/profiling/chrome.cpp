// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file chrome.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambase/profiling/chrome.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <variant>

u32 chrome_pid = u32_max;

f64 time_offset = 0.0;

void shambase::profiling::chrome::set_time_offset(f64 offset) { time_offset = offset; }

void shambase::profiling::chrome::set_chrome_pid(u32 pid) { chrome_pid = pid; }

namespace {

    f64 to_prof_time(f64 in) { return (in - time_offset) * 1e6; };
    f64 to_prof_delta(f64 in) { return (in) * 1e6; };

    struct ChromeEvent {

        struct EventStart {
            std::string name;
            std::string category_name;
            f64 t_start;
            u64 pid;
            u64 tid;

            std::string to_string() const {
                return shambase::format(
                    "{{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"B\", \"ts\": {}, \"pid\": {}, "
                    "\"tid\": "
                    "{}}},\n",
                    name,
                    category_name,
                    to_prof_time(t_start),
                    pid,
                    tid);
            }
        };

        struct EventEnd {
            std::string name;
            std::string category_name;
            f64 tend;
            u64 pid;
            u64 tid;

            std::string to_string() const {
                return shambase::format(
                    "{{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"E\", \"ts\": {}, \"pid\": {}, "
                    "\"tid\": "
                    "{}}},\n",
                    name,
                    category_name,
                    to_prof_time(tend),
                    pid,
                    tid);
            }
        };

        struct EventComplete {
            std::string name;
            std::string category_name;
            f64 t_start;
            f64 tend;
            u64 pid;
            u64 tid;

            std::string to_string() const {
                return shambase::format(
                    "{{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"X\", \"ts\": {}, \"dur\": {}, "
                    "\"pid\": {}, "
                    "\"tid\": "
                    "{}}},\n",
                    name,
                    category_name,
                    to_prof_time(t_start),
                    to_prof_delta(tend - t_start),
                    pid,
                    tid);
            }
        };

        struct CounterEvent {
            std::string name;
            f64 t;
            f64 val;
            u64 pid;

            std::string to_string() const {
                return shambase::format(
                    "{{ \"pid\": {}, \"ph\": \"C\", \"ts\":  {}, \"args\": {{\"{}\": {:e} }} }},\n",
                    pid,
                    to_prof_time(t),
                    name,
                    val);
            }
        };

        using var_t = std::variant<EventStart, EventEnd, EventComplete, CounterEvent>;
        var_t var;

        std::string to_string() const {
            return std::visit(
                [](auto &&v) {
                    return v.to_string();
                },
                var);
        }

        void change_pid(u64 new_pid) {
            std::visit(
                [&](auto &&v) {
                    v.pid = new_pid;
                },
                var);
        }
    };

    struct EventStorage {

        std::vector<ChromeEvent> events = {};
        std::string filename_prefix     = "shamrock_chrome_trace";
        std::unique_ptr<std::ofstream> stream;

        EventStorage(std::string filename) : filename_prefix(filename) {}

        void flush_events() {

            if (chrome_pid == u32_max) {
                return;
            }

            if (!stream) {
                std::string filename = filename_prefix + "_" + std::to_string(chrome_pid) + ".json";
                stream               = std::make_unique<std::ofstream>(filename);
            }

            for (auto &e : events) {
                e.change_pid(chrome_pid);
                auto event = e.to_string();
                *stream << event;
            }

            events.clear();
        }

        void register_event(ChromeEvent event) {
            events.push_back(event);
            flush_events();
        }

        void register_event(ChromeEvent::EventStart event) { register_event(ChromeEvent{event}); }

        void register_event(ChromeEvent::EventEnd event) { register_event(ChromeEvent{event}); }

        void register_event(ChromeEvent::EventComplete event) {
            register_event(ChromeEvent{event});
        }

        void register_event(ChromeEvent::CounterEvent event) { register_event(ChromeEvent{event}); }
    };

    auto get_prof_prefix = []() {
        const char *val = std::getenv("SHAM_PROF_PREFIX");
        if (val != nullptr) {
            return std::string(val);
        }
        return std::string("shamrock_profile");
    };

    std::string prof_prefix = get_prof_prefix();
    EventStorage event_storage{prof_prefix};

} // namespace

void shambase::profiling::chrome::register_event_start(
    const std::string &name, const std::string &category_name, f64 t_start, u64 pid, u64 tid) {

    event_storage.register_event(ChromeEvent::EventStart{name, category_name, t_start, pid, tid});
}

void shambase::profiling::chrome::register_event_end(
    const std::string &name, const std::string &category_name, f64 tend, u64 pid, u64 tid) {
    event_storage.register_event(ChromeEvent::EventEnd{name, category_name, tend, pid, tid});
}

void shambase::profiling::chrome::register_event_complete(
    const std::string &name,
    const std::string &category_name,
    f64 t_start,
    f64 tend,
    u64 pid,
    u64 tid) {

    event_storage.register_event(
        ChromeEvent::EventComplete{name, category_name, t_start, tend, pid, tid});
}

void shambase::profiling::chrome::register_metadata_thread_name(
    u64 pid, u64 tid, const std::string &name) {
    // register_event("");
}

void shambase::profiling::chrome::register_counter_val(
    u64 pid, f64 t, const std::string &name, f64 val) {

    event_storage.register_event(ChromeEvent::CounterEvent{name, t, val, pid});
}

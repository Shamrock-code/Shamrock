// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file gpu_core_timeline.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Antoine Richermoz (antoine.richermoz@inria.fr)
 *
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include "shambase/numeric_limits.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/intrinsics.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include <shambackends/sycl.hpp>
#include <unordered_map>
#include <vector>

#if __has_include(<nlohmann/json.hpp>)
    #include "nlohmann/json.hpp"
#endif

namespace sham {

    /// @brief A timeline event for the gpu core timeline
    struct TimelineEvent {
        unsigned long long start;
        unsigned long long first_end;
        unsigned long long last_end;
        uint lane;
        uint color;
    };

} // namespace sham

#if __has_include(<nlohmann/json.hpp>)

NLOHMANN_JSON_NAMESPACE_BEGIN
template<>
struct adl_serializer<sham::TimelineEvent> {
    static void to_json(json &j, const sham::TimelineEvent &e) {
        j
            = {{"start", e.start},
               {"first_end", e.first_end},
               {"last_end", e.last_end},
               {"color", e.color},
               {"lane", e.lane}};
    }
};
NLOHMANN_JSON_NAMESPACE_END
#endif

namespace sham {

    /**
     * @brief This class implement the GPU core timeline tool from the original algorithm of
     * A. Richermoz, F. Neyret 2024
     *
     * This is a utility to profile the execution of kernels on a GPU. It
     * provides an interface to extract the execution timeline of each work-groups.
     *
     * @note see the test file gpu_core_timelineTest.cpp for example usage
     *
     */
    class gpu_core_timeline_profilier {
        sham::DeviceScheduler_ptr dev_sched;
        sham::DeviceBuffer<u64> frame_start_clock;

        sham::DeviceBuffer<TimelineEvent> events;
        sham::DeviceBuffer<u64> event_count;

        public:
        /// CTOR
        gpu_core_timeline_profilier(sham::DeviceScheduler_ptr dev_sched, u32 max_event_count)
            : dev_sched(dev_sched), frame_start_clock(sham::DeviceBuffer<u64>(1, dev_sched)),
              events(max_event_count, dev_sched), event_count(1, dev_sched) {
            event_count.set_val_at_idx(0, 0);
            is_available_on_device();
        }

        /**
         * @brief Check if gpu_core_timeline_profilier is available on the device
         *
         * This function checks if the current device supports both
         * sham::intrisics::get_device_clock and sham::intrisics::get_sm_id.
         * If not, a warning message is logged.
         *
         * This function is lazy, it will only check if the function is available on the first call.
         *
         * @return true if gpu_core_timeline_profilier is available, false otherwise
         */
        inline bool is_available_on_device() {

            static std::unordered_map<DeviceScheduler *, bool> cache;
            auto it = cache.find(dev_sched.get());
            if (it == cache.end()) {

                sham::DeviceBuffer<u64> tmp(1, dev_sched);

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{},
                    sham::MultiRef{tmp},
                    1,
                    [](u32 i, u64 *out) {
#if defined(SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE)                                         \
    && defined(SHAMROCK_INTRISICS_GET_SMID_AVAILABLE)
                        *out = 1;
#else
                        *out = 0;
#endif
                    });

                cache[dev_sched.get()] = tmp.get_val_at_idx(0);

                if (!cache[dev_sched.get()]) {
                    logger::warn_ln(
                        "Backend", "gpu_core_timeline_profilier is not available on the device");
                }
            }

            return cache[dev_sched.get()];
        }

        // base clock val

        /**
         * @brief Recover the current device time in the frame_start_clock buffer
         */
        void setFrameStartClock() {
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{frame_start_clock},
                1,
                [](u32 i, u64 *clock) {
#ifdef SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE
                    *clock = sham::get_device_clock();
#else
                    *clock = 0;
#endif
                });
        }

        inline u64 get_base_clock_value() { return frame_start_clock.get_val_at_idx(0); }

        struct local_access_t {
            sycl::local_accessor<uint> _index;
            sycl::local_accessor<bool> _valid;

            local_access_t(sycl::handler &cgh) : _index(1, cgh), _valid(1, cgh) {}
        };

        // Kernel access section
        struct acc {
            TimelineEvent *events;
            u64 *event_count;
            u64 max_event_count;

            /**
             * @brief Initialize a timeline event
             *
             * This function must be called at the beginning of the kernel to register a new
             * timeline event. The function will only be executed by one thread of the work-group,
             * and will be synchronized with the other threads of the work-group.
             *
             * This function will allocate a new timeline event if there is still space available in
             * the buffer. If the buffer is full, the function will do nothing.
             *
             * The function will also set the start time of the timeline event to the current device
             * clock.
             *
             * @param[in] item The sycl::nd_item representing the current work-group.
             * @param[in] acc The local accessor for the current work-group.
             */
            inline void
            init_timeline_event(sycl::nd_item<1> item, const local_access_t &acc) const {
                if (item.get_local_id(0) == 0) {
                    sycl::atomic_ref<
                        u64,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        ev_cnt_ref(event_count[0]);

                    acc._index[0] = ev_cnt_ref.fetch_add(1_u64);
                    acc._valid[0] = acc._index[0] < max_event_count;

                    if (acc._valid[0]) {
#ifdef SHAMROCK_INTRISICS_GET_SMID_AVAILABLE
                        events[acc._index[0]] = {u64_max, u64_max, 0, sham::get_sm_id(), 0};
#else
                        events[acc._index[0]] = {u64_max, u64_max, 0, 0, 0};
#endif
                    }
                }
                item.barrier(); // equivalent to __syncthreads
            }

            /**
             * @brief Start a timeline event
             *
             * This function must be called at the start of the kernel to register to get the start
             * time of the worker thread.
             *
             * @param[in] acc The local accessor for the current work-group.
             */
            inline void start_timeline_event(const local_access_t &acc) const {
                if (acc._valid[0]) {

                    sycl::atomic_ref<
                        unsigned long long,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        start_val(events[acc._index[0]].start);

                    using ull = unsigned long long;

#ifdef SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE
                    ull clock = sham::get_device_clock();
#else
                    ull clock = 0;
#endif

                    start_val.fetch_min(clock);
                }
            }

            /**
             * @brief Finish a timeline event
             *
             * This function must be called at the end of the kernel to register the resulting
             * events.
             *
             * @param[in] acc The local accessor for the current work-group.
             */
            inline void end_timeline_event(const local_access_t &acc) const {
                if (acc._valid[0]) {
                    sycl::atomic_ref<
                        unsigned long long,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        first_end(events[acc._index[0]].first_end);

                    sycl::atomic_ref<
                        unsigned long long,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        last_end(events[acc._index[0]].last_end);

                    using ull = unsigned long long;

#ifdef SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE
                    ull clock = sham::get_device_clock();
#else
                    ull clock = 0;
#endif

                    first_end.fetch_min(clock);
                    last_end.fetch_max(clock);
                }
            }
        };

        /**
         * @brief Get a write access to the timeline events and the event count.
         *
         * @param deps The event list to wait on.
         * @return A structure that contains a write access to the timeline events,
         *         the write access to the event count and the size of the timeline events.
         */
        inline acc get_write_access(sham::EventList &deps) {
            return {
                events.get_write_access(deps),
                event_count.get_write_access(deps),
                events.get_size()};
        }

        /**
         * Completes the event state of the timeline events and the event count.
         * This function is necessary to ensure that all events are properly
         * registered after a kernel.
         *
         * @param e The event to wait on.
         */
        inline void complete_event_state(sycl::event e) {
            events.complete_event_state(e);
            event_count.complete_event_state(e);
        }

#if __has_include(<nlohmann/json.hpp>)
        /**
         * Dumps the timeline events to a JSON file.
         *
         * This function is only available if <nlohmann/json.hpp> is included.
         *
         * @param filename The name of the file to dump the timeline events.
         *
         * The dumped JSON will have the following structure:
         * <pre>
         * [
         *     {
         *         "start": <start value>,
         *         "first_end": <first end value>,
         *         "last_end": <last end value>,
         *         "name": "<name>"
         *     },
         *     ...
         * ]
         * </pre>
         */
        inline void dump_to_file(const std::string &filename) {

            u32 sz = event_count.get_val_at_idx(0);

            std::cout << "dumping to " << filename << " size = " << sz << std::endl;

            std::vector<TimelineEvent> events = this->events.copy_to_stdvec_idx_range(0, sz);

            u64 base_clock = get_base_clock_value();

            for (auto &t : events) {
                t.start -= base_clock;
                t.first_end -= base_clock;
                t.last_end -= base_clock;
            }

            std::ofstream file(filename);
            file << nlohmann::json(events).dump(4) << std::endl;
        }
#endif

        // inline void open_file(const std::string &filename) {
        //     std::string cmd = "python3 ../buildbot/gpu_core_timeline_read.py ";
        //     cmd += filename + " -b 4";
        //     std::system(cmd.c_str());
        // }
    };

} // namespace sham

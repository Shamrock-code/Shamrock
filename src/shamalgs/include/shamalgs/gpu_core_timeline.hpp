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
 * @brief main include file for memory algorithms
 *
 */

#include "shambase/numeric_limits.hpp"
#include "nlohmann/json.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include <shambackends/sycl.hpp>
#include <vector>

#ifdef __ACPP_ENABLE_CUDA_TARGET__
    #include <cuda/std/chrono>

ACPP_UNIVERSAL_TARGET
uint get_sm() {
    uint32_t ret;
    __acpp_if_target_cuda(asm("mov.u32 %0, %%smid;" : "=r"(ret)));
    return ret;
}
#else
uint get_sm() { return 0; }
#endif

ACPP_UNIVERSAL_TARGET inline u64 get_cuda_clock() {
    u64 out = 0;

    __acpp_if_target_cuda(
        out = cuda::std::chrono::high_resolution_clock::now().time_since_epoch().count(););

    return out;
}

namespace shamalgs {

    struct TimelineEvent {
        unsigned long long start;
        unsigned long long end;
        uint lane;
        uint color;
    };
} // namespace shamalgs
#if __has_include(<nlohmann/json.hpp>)
NLOHMANN_JSON_NAMESPACE_BEGIN
template<>
struct adl_serializer<shamalgs::TimelineEvent> {
    static void to_json(json &j, const shamalgs::TimelineEvent &e) {
        j = {{"start", e.start}, {"end", e.end}, {"color", e.color}, {"lane", e.lane}};
    }
};
NLOHMANN_JSON_NAMESPACE_END
#endif

namespace shamalgs {

    class gpu_core_timeline_profilier {
        sham::DeviceScheduler_ptr dev_sched;
        sham::DeviceBuffer<u64> frame_start_clock;

        sham::DeviceBuffer<TimelineEvent> events;
        sham::DeviceBuffer<u64> event_count;

        public:
        gpu_core_timeline_profilier(sham::DeviceScheduler_ptr dev_sched, u32 max_event_count)
            : dev_sched(dev_sched), frame_start_clock(sham::DeviceBuffer<u64>(1, dev_sched)),
              events(max_event_count, dev_sched), event_count(1, dev_sched) {}

        // base clock val

        void setFrameStartClock() {
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{frame_start_clock},
                1,
                [](u32 i, u64 *clock) {
                    *clock = get_cuda_clock();
                });
        }

        u64 get_base_clock_value() { return frame_start_clock.get_val_at_idx(0); }

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

            inline void
            init_timeline_event(sycl::nd_item<1> item, const local_access_t &acc) const {
                if (item.get_local_linear_id() == 0) {
                    sycl::atomic_ref<
                        u64,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        ev_cnt_ref(event_count[0]);

                    acc._index[0] = ev_cnt_ref.fetch_add(1_u64);
                    acc._valid[0] = acc._index[0] < max_event_count;

                    if (acc._valid[0]) {
                        events[acc._index[0]] = {u64_max, 0, get_sm(), 0};
                    }
                }
                item.barrier(); // equivalent to __syncthreads
            }

            inline void start_timeline_event(const local_access_t &acc) const {
                if (acc._valid[0]) {

                    sycl::atomic_ref<
                        unsigned long long,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        start_val(events[acc._index[0]].start);

                    using ull = unsigned long long;

                    start_val.fetch_min(ull(get_cuda_clock()));
                }
            }

            inline void end_timeline_event(const local_access_t &acc) const {
                if (acc._valid[0]) {
                    sycl::atomic_ref<
                        unsigned long long,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        end_val(events[acc._index[0]].end);

                    using ull = unsigned long long;

                    end_val.fetch_max(ull(get_cuda_clock()));
                }
            }
        };

        acc get_write_access(sham::EventList &deps) {
            return {
                events.get_write_access(deps),
                event_count.get_write_access(deps),
                events.get_size()};
        }

        void complete_event_state(sycl::event e) {
            events.complete_event_state(e);
            event_count.complete_event_state(e);
        }

        void dump_to_file(const std::string &filename) {

            std::vector<TimelineEvent> events
                = this->events.copy_to_stdvec_idx_range(0, event_count.get_val_at_idx(0));

            u64 base_clock = get_base_clock_value();

            for (auto &t : events) {
                t.start -= base_clock;
                t.end -= base_clock;
            }

            std::ofstream file(filename);
            file << nlohmann::json(events).dump(4) << std::endl;
        }
    };

} // namespace shamalgs

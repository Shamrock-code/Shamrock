// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FrameTimings.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per-frame CPU timings of shamrock_gui, reported as one JSON line by --bench.
 *
 */

#include "sham/gui/FrameTimings.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <numeric>

namespace sham::gui {

    double FrameTimings::wall() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    void FrameTimings::begin_frame() {
        t0 = wall();
        if (prev_frame_start >= 0)
            timings["frame"].push_back(t0 - prev_frame_start);
        prev_frame_start = t0;
    }

    void FrameTimings::mark_update() {
        t1 = wall();
        timings["update"].push_back(t1 - t0);
    }

    void FrameTimings::mark_ui() { timings["ui"].push_back(wall() - t1); }

    void FrameTimings::print(int warmup) const {
        std::printf("BENCH {\"impl\": \"cpp\"");
        for (const char *k : {"update", "ui", "frame"}) {
            const std::vector<double> &all = timings.at(k);
            std::vector<double> a(
                all.begin() + std::ptrdiff_t(std::min<size_t>(size_t(warmup), all.size())),
                all.end());
            for (double &v : a)
                v *= 1e3;
            std::sort(a.begin(), a.end());
            double mean
                = a.empty() ? 0 : std::accumulate(a.begin(), a.end(), 0.0) / double(a.size());
            auto pct = [&](double p) { // numpy's default (linear) percentile
                if (a.empty())
                    return 0.0;
                double idx = p / 100.0 * double(a.size() - 1);
                size_t lo = size_t(idx), hi = std::min(lo + 1, a.size() - 1);
                return a[lo] + (a[hi] - a[lo]) * (idx - double(lo));
            };
            std::printf(
                ", \"%s\": {\"mean_ms\": %.3f, \"median_ms\": %.3f, \"p95_ms\": %.3f}",
                k,
                mean,
                pct(50),
                pct(95));
        }
        std::printf("}\n");
    }

} // namespace sham::gui

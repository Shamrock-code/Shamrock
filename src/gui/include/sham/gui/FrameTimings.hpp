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
 * @file FrameTimings.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per-frame CPU timings of shamrock_gui, reported as one JSON line by --bench.
 *
 */

#include <map>
#include <string>
#include <vector>

namespace sham::gui {

    /// Per-frame CPU timings (in seconds), collected for --bench.
    ///
    /// Three series are recorded: "update" (data update), "ui" (building the UI) and "frame" (time
    /// between the starts of two consecutive frames, so it also covers rendering and the buffer
    /// swap).
    struct FrameTimings {
        /// samples in seconds, one entry per frame (except "frame": none for the first frame)
        std::map<std::string, std::vector<double>> timings{
            {"update", {}}, {"ui", {}}, {"frame", {}}};

        /// start of the previous frame, negative before the first frame
        double prev_frame_start = -1;

        /// Wall clock in seconds. Deliberately not GuiClock::now(), which follows the virtual 60
        /// fps clock in headless runs.
        static double wall();

        /// Start of a frame: records the "frame" sample (from the second frame on).
        void begin_frame();

        /// End of the data update: records the "update" sample.
        void mark_update();

        /// End of the UI construction: records the "ui" sample.
        void mark_ui();

        /// Print `BENCH {...}`, one JSON line with the mean, median and p95 (in ms) of each series,
        /// skipping the first `warmup` frames. Percentiles are linearly interpolated, as numpy's
        /// default.
        void print(int warmup) const;

        private:
        double t0 = 0; ///< start of the current frame
        double t1 = 0; ///< end of the data update of the current frame
    };

} // namespace sham::gui

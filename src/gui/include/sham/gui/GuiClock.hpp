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
 * @file GuiClock.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Time source of shamrock_gui: fixed 60 fps virtual clock or wall clock.
 *
 */

#include <chrono>

namespace sham::gui {

    /// Time source of the GUI: a fixed 60 fps virtual clock when deterministic, the wall clock
    /// otherwise.
    struct GuiClock {
        bool deterministic;
        long long frame_counter = 0; ///< frames completed so far

        explicit GuiClock(bool deterministic_) : deterministic(deterministic_) {}

        /// Mark the end of a frame.
        void end_frame() { frame_counter += 1; }

        /// Current time in seconds.
        double now() const {
            using namespace std::chrono;
            return deterministic ? double(frame_counter) / 60.0
                                 : duration<double>(steady_clock::now().time_since_epoch()).count();
        }
    };

} // namespace sham::gui

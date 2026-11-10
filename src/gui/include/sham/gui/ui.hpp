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
 * @file ui.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief UI scale state and logical -> physical pixel conversion. Drawing in logical pixels goes
 * through ScaledDrawList.hpp.
 *
 */

#include "imgui.h"

namespace sham::gui {

    inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

    /// UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); P() and S()
    /// convert them to physical pixels.
    struct UI {
        /// current scale (physical pixels per logical pixel)
        double scale = 1.0;

        /// main viewport position (scaling is done around it)
        ImVec2 origin{0, 0};

        /// logical point -> physical point
        ImVec2 P(ImVec2 p) const {
            const float s = float(scale);
            return ImVec2(origin.x + (p.x - origin.x) * s, origin.y + (p.y - origin.y) * s);
        }

        /// logical length -> physical length
        float S(double v) const { return float(v * scale); }
    };

} // namespace sham::gui

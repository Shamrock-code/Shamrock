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
 * @file screenshot.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Save the OpenGL framebuffer of shamrock_gui as a PNG.
 *
 */

#include <string>

namespace sham::gui {

    /// Save the current framebuffer (after rendering, before the buffer swap) as a PNG.
    void take_screenshot(const std::string &screenshot_path);

} // namespace sham::gui

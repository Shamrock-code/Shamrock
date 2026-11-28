// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file screenshot.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Save the OpenGL framebuffer of shamrock_gui as a PNG.
 *
 */

#include "sham/gui/screenshot.hpp"
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sham::gui {

    void take_screenshot(const std::string &screenshot_path) {
        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &fbw, &fbh);
        std::vector<uint8_t> px(size_t(fbw) * fbh * 4), flipped(px.size());
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, fbw, fbh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        for (int j = 0; j < fbh; ++j)
            std::memcpy(
                &flipped[size_t(j) * fbw * 4], &px[size_t(fbh - 1 - j) * fbw * 4], size_t(fbw) * 4);
        for (size_t k = 3; k < flipped.size(); k += 4)
            flipped[k] = 255;
        stbi_write_png(screenshot_path.c_str(), fbw, fbh, 4, flipped.data(), fbw * 4);
        std::printf("saved %s\n", screenshot_path.c_str());
    }

} // namespace sham::gui

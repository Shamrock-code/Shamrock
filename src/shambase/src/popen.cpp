// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file popen.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/popen.hpp"
#include <stdexcept>

namespace shambase {

    std::string popen_fetch_output(const char *command) {

        char buffer[128];
        std::string result;
        FILE *pipe = popen(command, "r");
        if (!pipe)
            throw_with_loc<std::runtime_error>("popen() failed!");

        try {
            while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
                result += buffer;
            }
        } catch (...) {

            int exit_status = pclose(pipe);
            if (exit_status != 0) {
                throw_with_loc<std::runtime_error>(
                    "Command failed with exit status " + std::to_string(exit_status));
            }

            throw;
        }

        int exit_status = pclose(pipe);
        if (exit_status != 0) {
            throw_with_loc<std::runtime_error>(
                "Command failed with exit status " + std::to_string(exit_status));
        }

        return result;
    }

} // namespace shambase

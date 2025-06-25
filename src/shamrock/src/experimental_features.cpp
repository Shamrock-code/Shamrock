// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file experimental_features.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamrock/experimental_features.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"

namespace {
    bool _env_allow_experimental_features
        = shamcmdopt::getenv_str_default_register(
              "SHAM_EXPERIMENTAL", "0", "Allow the use of experimental features")
          == "1";

    bool warning_printed = false;
} // namespace

namespace shamrock {
    bool are_experimental_features_allowed() {
        if (!warning_printed && _env_allow_experimental_features) {
            if (shamcomm::world_rank() == 0) {

                std::string color = shambase::term_colors::col8b_yellow();
                std::string reset = shambase::term_colors::reset();

                shamcomm::logs::raw_ln(
                    "\n" + color
                    + "---------------------------- WARNING ----------------------------" + reset
                    + "\n" + color + "Warning:" + reset + " Experimental features are enabled\n"
                    + color + "-----------------------------------------------------------------"
                    + reset);
            }
            warning_printed = true;
        }

        return _env_allow_experimental_features;
    }

} // namespace shamrock

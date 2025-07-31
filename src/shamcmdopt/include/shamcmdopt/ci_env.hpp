// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ci_env.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

namespace shamcmdopt {

    /**
     * @brief Check if the environment variable GITHUB_ACTIONS is set
     *
     * This is used to check if the code is run in a GitHub Actions CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_github_actions();

    /**
     * @brief Check if the environment variable TRAVIS is set
     *
     * This is used to check if the code is run in a Travis CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_travis();

    /**
     * @brief Check if the environment variable CIRCLECI is set
     *
     * This is used to check if the code is run in a CircleCI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_circle_ci();

    /**
     * @brief Check if the environment variable GITLAB_CI is set
     *
     * This is used to check if the code is run in a GitLab CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_gitlab_ci();

} // namespace shamcmdopt

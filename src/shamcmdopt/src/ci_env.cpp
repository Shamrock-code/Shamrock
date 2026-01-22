// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ci_env.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcmdopt/ci_env.hpp"
#include "shamcmdopt/env.hpp"

namespace {

    using namespace shamcmdopt;

    bool GITHUB_ACTIONS = bool(getenv_str("GITHUB_ACTIONS"));
    bool TRAVIS         = bool(getenv_str("TRAVIS"));
    bool CIRCLECI       = bool(getenv_str("CIRCLECI"));
    bool GITLAB_CI      = bool(getenv_str("GITLAB_CI"));

} // namespace

namespace shamcmdopt {

    /**
     * @brief Check if the environment variable GITHUB_ACTIONS is set
     *
     * This is used to check if the code is run in a GitHub Actions CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_github_actions() { return GITHUB_ACTIONS; }

    /**
     * @brief Check if the environment variable TRAVIS is set
     *
     * This is used to check if the code is run in a Travis CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_travis() { return TRAVIS; }

    /**
     * @brief Check if the environment variable CIRCLECI is set
     *
     * This is used to check if the code is run in a CircleCI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_circle_ci() { return CIRCLECI; }

    /**
     * @brief Check if the environment variable GITLAB_CI is set
     *
     * This is used to check if the code is run in a GitLab CI job.
     *
     * @return true if the environment variable is set, false otherwise
     */
    bool is_ci_gitlab_ci() { return GITLAB_CI; }

} // namespace shamcmdopt

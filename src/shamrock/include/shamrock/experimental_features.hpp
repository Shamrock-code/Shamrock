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
 * @file experimental_features.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include <stdexcept>

namespace shamrock {

    /// Allow the use of experimental features
    bool are_experimental_features_allowed();

    /// Allow the use of experimental features
    void enable_experimental_features();

    /// Check if experimental features are enabled, if not throw with the given message
    inline void experimental_feature_check(
        const std::string &message, SourceLocation loc = SourceLocation{}) {

        if (!are_experimental_features_allowed()) {
            throw shambase::make_except_with_loc<std::runtime_error>(message, loc);
        }
    }

    /// Utility that can be used directly through it's constructor or that can be set as a base of a
    /// class to mark it as experimental
    class ExperimentalClassMarker {
        public:
        ExperimentalClassMarker(
            const std::string &custom_message, SourceLocation loc = SourceLocation{}) {
            experimental_feature_check(custom_message, loc);
        }

        ExperimentalClassMarker(SourceLocation loc = SourceLocation{})
            : shamrock::ExperimentalClassMarker(
                  "You are trying to use experimental features without having enabled", loc) {}
    };

} // namespace shamrock

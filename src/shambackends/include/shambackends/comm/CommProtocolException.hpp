// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CommProtocolException.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <exception>
#include <string>

namespace shamsys::comm {

    /**
     * @brief Exception type for the NodeInstance
     */
    class CommProtocolException : public std::exception {
        public:
        explicit CommProtocolException(const char *message) : msg_(message) {}

        explicit CommProtocolException(const std::string &message) : msg_(message) {}

        ~CommProtocolException() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

        protected:
        std::string msg_;
    };

} // namespace shamsys::comm

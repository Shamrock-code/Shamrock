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
 * @file sycl_handler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief header file to manage sycl
 */

#include "shambase/exception.hpp"
#include <sycl/sycl.hpp>

class ShamrockSyclException : public std::exception {
    public:
    explicit ShamrockSyclException(const char *message) : msg_(message) {}

    explicit ShamrockSyclException(const std::string &message) : msg_(message) {}

    ~ShamrockSyclException() noexcept override = default;

    [[nodiscard]]
    const char *what() const noexcept override {
        return msg_.c_str();
    }

    protected:
    std::string msg_;
};

namespace sycl_handler {

    // void init();
    //
    // sycl::queue &get_compute_queue();
    // sycl::queue &get_alt_queue();

} // namespace sycl_handler

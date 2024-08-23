// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CommProtocolException.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

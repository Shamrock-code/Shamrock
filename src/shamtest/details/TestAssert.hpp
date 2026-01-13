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
 * @file TestAssert.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_int.hpp"

namespace shamtest::details {

    /// A test assertion
    struct TestAssert {
        bool value;          ///< Value of the assert
        std::string name;    ///< Name of the assert
        std::string comment; ///< Comment attached to the assert

        /// Serialize the assertion in JSON
        std::string serialize_json();

        /// Serialize the assertion in binary format
        void serialize(std::basic_stringstream<byte> &stream);

        /// DeSerialize the assertion from binary format
        static TestAssert deserialize(std::basic_stringstream<byte> &reader);
    };

} // namespace shamtest::details

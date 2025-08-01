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
 * @file DataNode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file hold the definitions for a test DataNode
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <string>
#include <vector>

namespace shamtest::details {

    /// Data node generated by the test
    struct DataNode {
        std::string name;      ///< Name of the data node
        std::vector<f64> data; ///< Held data

        /// Serialize the assertion in JSON
        std::string serialize_json();
        /// Serialize the assertion in binary format
        void serialize(std::basic_stringstream<byte> &stream);
        /// DeSerialize the assertion from binary format
        static DataNode deserialize(std::basic_stringstream<byte> &reader);
    };
} // namespace shamtest::details

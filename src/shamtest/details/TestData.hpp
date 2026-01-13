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
 * @file TestData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtest/details/DataNode.hpp"

namespace shamtest::details {

    ///< Test data
    struct TestData {
        std::string dataset_name;      ///< Name of the dataset
        std::vector<DataNode> dataset; ///< Dataset

        /// Add some data to the dataset
        inline void add_data(std::string name, const std::vector<f64> &v) {
            std::vector<f64> new_vec;
            for (f64 f : v) {
                new_vec.push_back(f);
            }
            dataset.push_back(DataNode{std::move(name), std::move(new_vec)});
        }

        /// Serialize the assertion in JSON
        std::string serialize_json();
        /// Serialize the assertion in binary format
        void serialize(std::basic_stringstream<byte> &stream);
        /// DeSerialize the assertion from binary format
        static TestData deserialize(std::basic_stringstream<byte> &stream);
    };

} // namespace shamtest::details

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TestData.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

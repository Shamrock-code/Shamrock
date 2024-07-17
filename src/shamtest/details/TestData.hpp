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

    struct TestData {
        std::string dataset_name;
        std::vector<DataNode> dataset;

        inline void add_data(std::string name, const std::vector<f64> &v) {
            std::vector<f64> new_vec;
            for (f64 f : v) {
                new_vec.push_back(f);
            }
            dataset.push_back(DataNode{std::move(name), std::move(new_vec)});
        }

        std::string serialize_json();

        void serialize(std::basic_stringstream<byte> &stream);
        static TestData deserialize(std::basic_stringstream<byte> &stream);
    };

} // namespace shamtest::details

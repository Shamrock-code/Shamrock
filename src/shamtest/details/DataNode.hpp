// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DataNode.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file hold the definitions for a test DataNode
 */

#include "aliases.hpp"

namespace shamtest::details {
    struct DataNode {
        std::string name;
        std::vector<f64> data;

        std::string serialize_json();

        void serialize(std::basic_stringstream<byte> &stream);
        static DataNode deserialize(std::basic_stringstream<byte> &reader);
    };
} // namespace shamtest::details
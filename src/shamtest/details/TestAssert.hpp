// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TestAssert.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shambase/SourceLocation.hpp"

namespace shamtest::details {

    struct TestAssert {
        bool value;
        std::string name;
        std::string comment;

        std::string serialize_json();
        void serialize(std::basic_stringstream<byte> &stream);
        static TestAssert deserialize(std::basic_stringstream<byte> &reader);

    };

} // namespace shamtest::details
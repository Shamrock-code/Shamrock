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
 * @file type_name_info.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/type_traits.hpp"
#include <string>

namespace shambase {

    template<class T>
    struct TypeNameInfo {
        static_assert(
            always_false_v<T>,
            "This name was not specialized, please either include the file with the definiton or "
            "add a new one");
        inline static const std::string name = "";
    };

    template<class T>
    std::string get_type_name() {
        return TypeNameInfo<T>::name;
    }

} // namespace shambase

namespace shambase {

    template<>
    struct TypeNameInfo<i64> {
        inline static const std::string name = "i64";
    };
    template<>
    struct TypeNameInfo<i32> {
        inline static const std::string name = "i32";
    };
    template<>
    struct TypeNameInfo<i16> {
        inline static const std::string name = "i16";
    };
    template<>
    struct TypeNameInfo<i8> {
        inline static const std::string name = "i8";
    };
    template<>
    struct TypeNameInfo<u64> {
        inline static const std::string name = "u64";
    };
    template<>
    struct TypeNameInfo<u32> {
        inline static const std::string name = "u32";
    };
    template<>
    struct TypeNameInfo<u16> {
        inline static const std::string name = "u16";
    };
    template<>
    struct TypeNameInfo<u8> {
        inline static const std::string name = "u8";
    };

    template<>
    struct TypeNameInfo<f64> {
        inline static const std::string name = "f64";
    };
    template<>
    struct TypeNameInfo<f32> {
        inline static const std::string name = "f32";
    };

} // namespace shambase

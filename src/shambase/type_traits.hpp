// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include <climits>
#include <type_traits>

namespace shambase {

    template<class T>
    inline constexpr u64 bitsizeof = sizeof(T) * CHAR_BIT;

    template<typename T, int num>
    struct has_bitlen {
        static constexpr bool value = bitsizeof<T> == num;
    };

    template<typename T, int num>
    inline constexpr bool has_bitlen_v = has_bitlen<T, num>::value;

    inline constexpr bool is_valid_sycl_vec_size(int N) {
        return N == 2 || N == 3 || N == 4 || N == 8 || N == 16;
    }

    template<class T>
    inline constexpr bool is_valid_sycl_base_type =
        std::is_same_v<T, i64> || std::is_same_v<T, i32> || std::is_same_v<T, i16> ||
        std::is_same_v<T, i8> || std::is_same_v<T, u64> || std::is_same_v<T, u32> ||
        std::is_same_v<T, u16> || std::is_same_v<T, u8> || std::is_same_v<T, f16> ||
        std::is_same_v<T, f32> || std::is_same_v<T, f64>;

    template<typename T>
    class has_operator_self_geq {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested >= tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_leq {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested <= tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_greater_than {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested > tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_less_than {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested < tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

} // namespace shambase
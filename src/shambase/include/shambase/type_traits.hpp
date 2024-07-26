// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file type_traits.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Traits for C++ types
 *
 * This file contains utility traits classes for C++ types.
 *
 */

#include "shambase/aliases_int.hpp"
#include <type_traits>
#include <climits>

namespace shambase {

    /**
     * @brief Number of bits in a type T
     *
     * @tparam T type for which the number of bits is computed
     */
    template<class T>
    inline constexpr u64 bitsizeof = sizeof(T) * CHAR_BIT;

    /**
     * @brief Check if a type has a certain number of bits
     *
     * This struct template provides a static constexpr bool member `value` which
     * is true if the type `T` has `num` bits.
     *
     * @tparam T type to check
     * @tparam num number of bits
     */
    template<typename T, int num>
    struct has_bitlen {
        /// True if bitlen of T is equal to num
        static constexpr bool value = bitsizeof<T> == num;
    };

    /**
     * @brief Helper variable template for has_bitlen
     *
     * This variable template is true if `T` has `num` bits.
     *
     * @tparam T type to check
     * @tparam num number of bits
     */
    template<typename T, int num>
    inline constexpr bool has_bitlen_v = has_bitlen<T, num>::value;

    /**
     * @brief Checks if the type `T` has an operator ">=" defined for
     *        self-comparison.
     *
     * @tparam T type to check
     */
    template<typename T>
    class has_operator_self_geq {
        // Type used to return true if the operator is found
        using found = std::true_type;
        // Type used to return false if the operator is not found
        using not_found = std::false_type;

        /**
         * @brief Test if the operator is defined for type `A`
         *
         * @tparam A type to check
         * @param tested dummy parameter to select the right overload
         * @return `found()` if the operator is defined, `not_found()` otherwise
         */
        template<typename A>
        static auto test(A tested) -> decltype(tested >= tested, found());
        /**
         * @brief Fallback overload for any type `A`
         *
         * @tparam A type to check
         * @param ... dummy parameter to select the right overload
         * @return `not_found()`
         */
        template<typename A>
        static not_found test(...);

        public:
        /**
         * @brief Static constexpr bool member that is true if the operator `>=` is
         *        defined for type `T`.
         */
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    /**
     * @brief Checks if the type `T` has an operator "<=" defined for
     *        self-comparison.
     *
     * @tparam T type to check
     */
    template<typename T>
    class has_operator_self_leq {
        // Type used to return true if the operator is found
        using found = std::true_type;
        // Type used to return false if the operator is not found
        using not_found = std::false_type;

        /**
         * @brief Test if the operator is defined for type `A`
         *
         * @tparam A type to check
         * @param tested dummy parameter to select the right overload
         * @return `found()` if the operator is defined, `not_found()` otherwise
         */
        template<typename A>
        static auto test(A tested) -> decltype(tested <= tested, found());
        /**
         * @brief Fallback overload for any type `A`
         *
         * @tparam A type to check
         * @param ... dummy parameter to select the right overload
         * @return `not_found()`
         */
        template<typename A>
        static not_found test(...);

        public:
        /**
         * @brief Static constexpr bool member that is true if the operator `<=` is
         *        defined for type `T`.
         */
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    /**
     * @brief Checks if the type `T` has an operator ">" defined for
     *        self-comparison.
     *
     * @tparam T type to check
     */
    template<typename T>
    class has_operator_self_greater_than {
        // Type used to return true if the operator is found
        using found = std::true_type;
        // Type used to return false if the operator is not found
        using not_found = std::false_type;

        /**
         * @brief Test if the operator is defined for type `A`
         *
         * @tparam A type to check
         * @param tested dummy parameter to select the right overload
         * @return `found()` if the operator is defined, `not_found()` otherwise
         */
        template<typename A>
        static auto test(A tested) -> decltype(tested > tested, found());
        /**
         * @brief Fallback overload for any type `A`
         *
         * @tparam A type to check
         * @param ... dummy parameter to select the right overload
         * @return `not_found()`
         */
        template<typename A>
        static not_found test(...);

        public:
        /**
         * @brief Static constexpr bool member that is true if the operator `>` is
         *        defined for type `T`.
         */
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    /**
     * @brief Checks if the type `T` has an operator "<" defined for
     *        self-comparison.
     *
     * @tparam T type to check
     */
    template<typename T>
    class has_operator_self_less_than {
        // Type used to return true if the operator is found
        using found = std::true_type;
        // Type used to return false if the operator is not found
        using not_found = std::false_type;

        /**
         * @brief Test if the operator is defined for type `A`
         *
         * @tparam A type to check
         * @param tested dummy parameter to select the right overload
         * @return `found()` if the operator is defined, `not_found()` otherwise
         */
        template<typename A>
        static auto test(A tested) -> decltype(tested < tested, found());
        /**
         * @brief Fallback overload for any type `A`
         *
         * @tparam A type to check
         * @param ... dummy parameter to select the right overload
         * @return `not_found()`
         */
        template<typename A>
        static not_found test(...);

        public:
        /**
         * @brief Static constexpr bool member that is true if the operator `<` is
         *        defined for type `T`.
         */
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    /**
     * @brief Helper variable template that is always false.
     * Especially usefull to perform static asserts based on templates
     * @code{.cpp}
     * if constexpr(constexpr_condition(var)){
     *     ......
     * }else {
     *     static_assert(shambase::always_false_v<decltype(var)>, "non-exhaustive visitor!");
     * }
     * @endcode
     * @tparam T dummy parameter to select the right overload
     */
    template<class>
    inline constexpr bool always_false_v = false;

} // namespace shambase

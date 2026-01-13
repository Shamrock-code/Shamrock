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
 * @file type_traits.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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

    /**
     * @brief Helper variable template that is used to associate a type to a boolean value in the
     * compile logs.
     *
     * @tparam val boolean value to transport
     * @tparam type The dummy type to be associated with the boolean value
     */
    template<bool val, class type>
    inline constexpr bool typed_false_v = val;

    /**
     * @brief Check if a lambda or a function has the correct signature.
     *
     * The function does nothing at runtime, but at compile time it checks if the
     * given lambda or function has the correct signature. If the signature
     * is wrong, the static assert will fail and the correct signature will be
     * indicated in the signature of `always_false_v<...>`.
     *
     * @warning This does not work with capturing lambdas
     *
     * @code {.cpp}
     * auto func = [](int a, int b) -> int { return a + b; };
     * shambase::check_functor_nocapture<int(int, int)>(func);
     * @endcode
     */
    template<class signature, class Func>
    constexpr void check_functor_nocapture(Func &&func) {
        constexpr bool result = std::is_assignable<signature *&, decltype(func)>::value;

        static_assert(
            typed_false_v<result, signature>,
            "The lambda signature is incorrect, the correct type is indicated in the signature of "
            "typed_false_v<...>.");
    }

    /**
     * @brief Check if a callable object has the correct deduced signature.
     *
     * This function deduces the signature of the callable object using the provided
     * return type and argument types, then checks if the callable matches this signature
     * using `check_functor`.
     *
     * @tparam RetType The return type of the callable.
     * @tparam Targ The types of the arguments the callable takes.
     * @tparam Func The type of the callable object to check.
     * @param func The callable object to be checked.
     */
    template<class RetType, class... Targ, class Func>
    constexpr void check_functor_nocapture_deduce(Func &&func, Targ...) {
        using signature = RetType(std::remove_reference_t<Targ>...);
        check_functor_nocapture<signature>(func);
    }

    /**
     * @brief Check if a callable object has the correct deduced signature.
     *
     * This function deduces the signature of the callable object using the provided
     * return type and argument types, then checks if the callable matches this signature
     * using `check_functor` and also checks if the return type matches the deduced return type.
     *
     * Exemple :
     * @code {.cpp}
     * int x = 0;
     * auto func = [x](int a, int b) -> int { return a + b; };
     * shambase::check_functor_signature<int(int, int)>(func);
     * @endcode
     *
     * @note This function works also for lambda with captures
     *
     * @tparam RetType The return type of the callable.
     * @tparam Targ The types of the arguments the callable takes.
     * @tparam Func The type of the callable object to check.
     * @param func The callable object to be checked.
     */
    template<class RetType, class... Targ, class Func>
    constexpr void check_functor_signature(Func &&func) {

        using signature = RetType(std::remove_reference_t<Targ>...);

        constexpr bool result_call = std::is_invocable_v<decltype(func), Targ...>;
        static_assert(
            typed_false_v<result_call, signature>,
            "The lambda signature is incorrect, the correct function signature is indicated in the "
            "signature of typed_false_v<...>.");

        using ret_t                 = decltype(func(std::declval<Targ>()...));
        constexpr bool result_ret_t = std::is_same_v<ret_t, RetType>;
        static_assert(
            typed_false_v<result_ret_t, signature>,
            "The lambda return type is incorrect, the correct function signature is indicated in "
            "the signature of typed_false_v<...>.");
    }

    /// variant of check_functor_signature that does not check the return type
    template<class... Targ, class Func>
    constexpr void check_functor_signature_noreturn(Func &&func) {

        using signature = void(std::remove_reference_t<Targ>...);

        constexpr bool result_call = std::is_invocable_v<decltype(func), Targ...>;
        static_assert(
            typed_false_v<result_call, signature>,
            "The lambda signature is incorrect, the correct function signature is indicated in the "
            "signature of typed_false_v<...>, aside for the return type.");
    }

    /**
     * @brief Check if a callable object has the correct deduced signature.
     *
     * This function deduces the signature of the callable object using the provided
     * return type and argument types, then checks if the callable matches this signature
     * using `check_functor_signature`.
     *
     * @tparam RetType The return type of the callable.
     * @tparam Targ The types of the arguments the callable takes.
     * @tparam Func The type of the callable object to check.
     * @param func The callable object to be checked.
     */
    template<class RetType, class... Targ, class Func>
    constexpr void check_functor_signature_deduce(Func &&func, Targ...) {
        check_functor_signature<RetType, Targ...>(func);
    }

    /// variant of check_functor_signature_deduce that does not check the return type
    template<class... Targ, class Func>
    constexpr void check_functor_signature_deduce_noreturn(Func &&func, Targ...) {
        check_functor_signature_noreturn<Targ...>(func);
    }

    /**
     * @brief variant of check_functor_signature_deduce that does not check the return type and
     * where some types can be specified manually
     *
     * For example when using a type with a reference, this is not properly deduced.
     * This functions allows to specify the type manually with the reference.
     *
     * @code {.cpp}
     * shambase::check_functor_signature_deduce_noreturn_add_t<sycl::handler&>(
     *   func,
     *   __acc_in...,
     *   __acc_in_out...,
     *   args...);
     * @endcode
     *
     * Here the function signature will be auto(sycl::handler&, <other types>)
     */
    template<class... Targ2, class... Targ, class Func>
    constexpr void check_functor_signature_deduce_noreturn_add_t(Func &&func, Targ...) {
        check_functor_signature_noreturn<Targ2..., Targ...>(func);
    }

} // namespace shambase

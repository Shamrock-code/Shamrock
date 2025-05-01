// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file memory.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include <memory>
#include <optional>
#include <utility>

namespace shambase {

    /**
     * @brief Store a value of type T in a byte buffer
     *
     * Store the given value of type T into the byte buffer at the given index.
     * The address is the starting address of the value to store in the buffer.
     *
     * @tparam T type of the value to store
     * @tparam AccU8 type of the byte buffer
     * @param acc byte buffer
     * @param idx_write index to store the value in the byte buffer
     * @param a value to store
     */
    template<class T, class AccU8>
    inline void store_u8(AccU8 &acc, u64 idx_write, T a) {
        constexpr u64 szT = sizeof(T);
        u8 *bytes         = (u8 *) &a;
#pragma unroll
        for (u64 i = 0; i < szT; i++) {
            acc[idx_write + i] = bytes[i];
        }
    }

    /**
     * @brief load a value of type T from a byte buffer
     *
     * @tparam T
     * @tparam AccU8
     * @param acc
     * @param ptr_load
     * @return T
     */
    template<class T, class AccU8>
    inline T load_u8(AccU8 &acc, u64 ptr_load) {
        constexpr u64 szT = sizeof(T);
        T ret;
        u8 *bytes = (u8 *) &ret;
#pragma unroll
        for (u64 i = 0; i < szT; i++) {
            bytes[i] = acc[ptr_load + i];
        }
        return ret;
    }

    /**
     * @brief pointer cast store the value @param a in the pointer
     *
     * @tparam T
     * @tparam TAcc
     * @param acc
     * @param a
     */
    template<class T, class TAcc>
    inline void store_conv(TAcc *acc, T a) {
        T *ptr = (T *) acc;
        *ptr   = a;
    }

    /**
     * @brief pointer cast load from a pointer
     *
     * @tparam T
     * @tparam TAcc
     * @param acc
     * @return T
     */
    template<class T, class TAcc>
    inline T load_conv(TAcc *acc) {
        T *ptr = (T *) acc;
        return *ptr;
    }

    /**
     * @brief Takes a std::unique_ptr and returns a reference to the object it holds.
     * It throws a std::runtime_error if the std::unique_ptr does not hold anything.
     *
     * @tparam T The type of the object held by the std::unique_ptr.
     * @param ptr A reference to the std::unique_ptr.
     * @param loc The source location where this function is called.
     * @return A reference to the object held by the std::unique_ptr.
     * @throws std::runtime_error If the std::unique_ptr does not hold anything.
     */
    template<class T>
    inline T &get_check_ref(const std::unique_ptr<T> &ptr, SourceLocation loc = SourceLocation()) {
        // Check if the std::unique_ptr holds anything
        if (!bool(ptr)) {
            // Throw a std::runtime_error with a descriptive message
            throw make_except_with_loc<std::runtime_error>("the ptr does not hold anything", loc);
        }
        // Return a reference to the object held by the std::unique_ptr
        return *ptr;
    }

    /**
     * @brief Takes a std::shared_ptr and returns a reference to the object it holds.
     * It throws a std::runtime_error if the std::shared_ptr does not hold anything.
     *
     * @tparam T The type of the object held by the std::shared_ptr.
     * @param ptr A reference to the std::shared_ptr.
     * @param loc The source location where this function is called.
     * @return A reference to the object held by the std::shared_ptr.
     * @throws std::runtime_error If the std::shared_ptr does not hold anything.
     */
    template<class T>
    inline T &get_check_ref(const std::shared_ptr<T> &ptr, SourceLocation loc = SourceLocation()) {
        // Check if the std::shared_ptr holds anything
        if (!bool(ptr)) {
            // Throw a std::runtime_error with a descriptive message
            throw make_except_with_loc<std::runtime_error>("the ptr does not hold anything", loc);
        }
        // Return a reference to the object held by the std::shared_ptr
        return *ptr;
    }

    /**
     * @brief Get reference to object held by the unique ptr, and throw if nothing is held
     *
     * @tparam T
     * @param ptr
     * @return T&
     */
    template<class T>
    inline const T &
    get_check_ref(const std::optional<T> &ptr, SourceLocation loc = SourceLocation()) {
        if (!bool(ptr)) {
            throw make_except_with_loc<std::runtime_error>("the ptr does not hold anything", loc);
        }
        return *ptr;
    }

    /// Same overload but for non const case
    template<class T>
    inline T &get_check_ref(std::optional<T> &ptr, SourceLocation loc = SourceLocation()) {
        if (!bool(ptr)) {
            throw make_except_with_loc<std::runtime_error>("the ptr does not hold anything", loc);
        }
        return *ptr;
    }

    /**
     * @brief Extracts the content out of an optional
     *
     * This function that takes an std::optional object and extracts the value from it.
     * If the optional is empty, it throws a runtime error. The extracted value is returned,
     * and the optional is left in an empty state.
     *
     * @see https://stackoverflow.com/questions/71980007/take-value-out-of-stdoptional
     *
     * @tparam T the type of the optional
     * @param o  reference to the optional object
     * @param loc The source location where this function is called.
     * @return T the value extracted from the optional
     * @throws std::runtime_error If the std::unique_ptr does not hold anything.
     */
    template<typename T>
    auto extract_value(std::optional<T> &o, SourceLocation loc = SourceLocation()) -> T {
        if (!bool(o)) {
            throw make_except_with_loc<std::runtime_error>(
                "the value cannot be extracted, as the optional is empty", loc);
        }
        return std::exchange(o, std::nullopt).value();
    }

    /**
     * @brief extract content out of unique_ptr
     *
     * @tparam T
     * @param o
     * @return T
     */
    template<typename T>
    auto extract_pointer(std::unique_ptr<T> &o, SourceLocation loc = SourceLocation()) -> T {
        if (!bool(o)) {
            throw make_except_with_loc<std::runtime_error>(
                "the value cannot be extracted, as the unique_ptr is empty", loc);
        }
        std::unique_ptr<T> tmp = std::exchange(o, {});
        return T(std::move(*tmp));
    }

    /**
     * @brief Convert a vector to an array of size n
     *
     * @tparam n The size of the array
     * @tparam T The type of the elements in the array
     * @param in The input vector
     * @return std::array<T, n> The converted array
     * @throw std::invalid_argument If the input vector size does not match n
     */
    template<int n, class T>
    inline std::array<T, n> convert_to_array(std::vector<T> &in) {
        if (in.size() != n) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "you've input values with the wrong size, input size = {}, wanted = {}",
                in.size(),
                n));
        }

        std::array<T, n> tmp;

        for (u32 i = 0; i < n; i++) {
            tmp[i] = in[i];
        }

        return tmp;
    }

} // namespace shambase

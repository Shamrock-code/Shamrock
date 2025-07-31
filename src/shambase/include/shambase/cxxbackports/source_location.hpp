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
 * @file source_location.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

/**
 * @class std::source_location
 * @brief Utility class to emulates
 * [std::source_location](https://en.cppreference.com/w/cpp/utility/source_location) class
 * introduced in c++20
 * This class provides information about the source location where it is used.
 *
 * @see https://en.cppreference.com/w/cpp/utility/source_location
 *
 * Exemple of usage :
 * \code{.cpp}
 * void log(const std::source_location location = std::source_location::current()){
 *     std::cout << "file: "
 *         << location.file_name() << '('
 *         << location.line() << ':'
 *         << location.column() << ") `"
 *         << location.function_name() << '\n';
 * }
 * \endcode
 */

/**
 * @fn std::source_location::current
 * @brief Returns the source location where this function is called.
 *
 * @return The source location.
 */

/**
 * @fn std::source_location::line
 * @brief Returns the line number of the source location.
 *
 * @return The line number.
 */

/**
 * @fn std::source_location::column
 * @brief Returns the column offset of the source location.
 *
 * @return The column offset.
 */

/**
 * @fn std::source_location::file_name
 * @brief Returns the file name of the source location.
 *
 * @return The file name.
 */

/**
 * @fn std::source_location::function_name
 * @brief Returns the function name of the source location.
 *
 * @return The function name.
 */

namespace std {

#if defined __has_builtin
    #if __has_builtin(__builtin_source_location)
        // if the builtin __builtin_source_location is available we can use the source_location
        // definition from libc++, otherwise we rebuild it using __builtin_FILE,
        // __builtin_FUNCTION,
        // __builtin_LINE and __builtin_COLUMN
        #define _INT_SOURCE_LOC_DEF
    #endif
#endif

#if defined(DOXYGEN)
    struct source_location {
        static constexpr source_location current() noexcept;
        constexpr source_location() noexcept = default;
        constexpr unsigned line() const noexcept;
        constexpr unsigned column() const noexcept;
        constexpr const char *file_name() const noexcept;
        constexpr const char *function_name() const noexcept;
    }
#else

    struct source_location {

    #ifndef _INT_SOURCE_LOC_DEF
        const char *_M_file_name     = "";
        const char *_M_function_name = "";
        unsigned _M_line             = 0;
        unsigned _M_column           = 0;

        public:
        static constexpr source_location current(

        #if defined __has_builtin
            #if __has_builtin(__builtin_FILE)
            const char *fileName = __builtin_FILE(),
            #else
            const char *fileName = "",
            #endif
            #if __has_builtin(__builtin_FUNCTION)
            const char *functionName = __builtin_FUNCTION(),
            #else
            const char *functionName = "",
            #endif
            #if __has_builtin(__builtin_LINE)
            const unsigned lineNumber = __builtin_LINE(),
            #else
            const unsigned lineNumber = 0,
            #endif
            #if __has_builtin(__builtin_COLUMN)
            const unsigned columnOffset = __builtin_COLUMN()
            #else
            const unsigned columnOffset = 0
            #endif
        #else

            const char *fileName        = "",
            const char *functionName    = "",
            const unsigned lineNumber   = 0,
            const unsigned columnOffset = 0
        #endif

                ) noexcept {
            source_location __sl;

            __sl._M_file_name     = fileName;
            __sl._M_function_name = functionName;
            __sl._M_line          = lineNumber;
            __sl._M_column        = columnOffset;
            return __sl;
        }

        constexpr source_location() noexcept = default;

        constexpr unsigned line() const noexcept { return _M_line; }
        constexpr unsigned column() const noexcept { return _M_column; }
        constexpr const char *file_name() const noexcept { return _M_file_name; }
        constexpr const char *function_name() const noexcept { return _M_function_name; }

    #else
        struct __impl {
            const char *_M_file_name;
            const char *_M_function_name;
            unsigned _M_line;
            unsigned _M_column;
        };

        const __impl *__ptr_ = nullptr;
        // GCC returns the type 'const void*' from the builtin, while clang returns
        // `const __impl*`. Per C++ [expr.const], casts from void* are not permitted
        // in constant evaluation, so we don't want to use `void*` as the argument
        // type unless the builtin returned that, anyhow, and the invalid cast is
        // unavoidable.

        using __bsl_ty = decltype(__builtin_source_location());

        public:
        // The defaulted __ptr argument is necessary so that the builtin is evaluated
        // in the context of the caller. An explicit value should never be provided.
        static constexpr source_location
        current(__bsl_ty __ptr = __builtin_source_location()) noexcept {
            source_location __sl;
            __sl.__ptr_ = static_cast<const __impl *>(__ptr);
            return __sl;
        }
        constexpr source_location() noexcept = default;

        constexpr unsigned line() const noexcept { return __ptr_ != nullptr ? __ptr_->_M_line : 0; }
        constexpr unsigned column() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_column : 0;
        }
        constexpr const char *file_name() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_file_name : "";
        }
        constexpr const char *function_name() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_function_name : "";
        }

        #undef _INT_SOURCE_LOC_DEF
    #endif
    };

#endif

} // namespace std

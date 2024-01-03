// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file source_location.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */
namespace std{

    #if defined __has_builtin
    #if __has_builtin(__builtin_source_location)
    #define _INT_SOURCE_LOC_DEF
    #endif
    #endif

    struct source_location {


    #ifndef _INT_SOURCE_LOC_DEF
        const char* _M_file_name = "";
        const char* _M_function_name = "";
        unsigned _M_line = 0;
        unsigned _M_column = 0;

        public:


        // The defaulted __ptr argument is necessary so that the builtin is evaluated
        // in the context of the caller. An explicit value should never be provided.
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

                const char *fileName = "",
                const char *functionName = "",
                const unsigned lineNumber = 0, const unsigned columnOffset = 0
            #endif

        ) noexcept {
            source_location __sl;

            __sl._M_file_name = fileName;
            __sl._M_function_name = functionName;
            __sl._M_line = lineNumber;
            __sl._M_column = columnOffset;
            return __sl;
        }

        constexpr source_location() noexcept = default;

        constexpr unsigned line() const noexcept {
            return _M_line;
        }
        constexpr unsigned column() const noexcept {
            return _M_column;
        }
        constexpr const char* file_name() const noexcept {
            return  _M_file_name;
        }
        constexpr const char* function_name() const noexcept {
            return _M_function_name;
        }


    #else
        struct __impl {
            const char* _M_file_name;
            const char* _M_function_name;
            unsigned _M_line;
            unsigned _M_column;
        };

        const __impl* __ptr_ = nullptr;
        // GCC returns the type 'const void*' from the builtin, while clang returns
        // `const __impl*`. Per C++ [expr.const], casts from void* are not permitted
        // in constant evaluation, so we don't want to use `void*` as the argument
        // type unless the builtin returned that, anyhow, and the invalid cast is
        // unavoidable.
        
        using __bsl_ty = decltype(__builtin_source_location());

        public:
        // The defaulted __ptr argument is necessary so that the builtin is evaluated
        // in the context of the caller. An explicit value should never be provided.
        static constexpr source_location current(__bsl_ty __ptr = __builtin_source_location()) noexcept {
            source_location __sl;
            __sl.__ptr_ = static_cast<const __impl*>(__ptr);
            return __sl;
        }
        constexpr source_location() noexcept = default;

        constexpr unsigned line() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_line : 0;
        }
        constexpr unsigned column() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_column : 0;
        }
        constexpr const char* file_name() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_file_name : "";
        }
        constexpr const char* function_name() const noexcept {
            return __ptr_ != nullptr ? __ptr_->_M_function_name : "";
        }

    #undef _INT_SOURCE_LOC_DEF
    #endif
    };

} // namespace std
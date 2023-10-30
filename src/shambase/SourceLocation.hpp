// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SourceLocation.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Source location utility
 * @date 2023-02-24
 */

#include "shambase/aliases_int.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

/**
 * @brief provide information about the source location
 * 
 * Exemple :
 * \code{.cpp}
 * SourceLocation loc = SourceLocation{};
 * \endcode
 */
struct SourceLocation {

    const char *fileName; /*!< The filename */
    const char *functionName;/*!< TThe name of the function */
    const u32 lineNumber;/*!< do i really need to explain this one XD */
    const u32 columnOffset;/*!< \see SourceLocation.lineNumber */

    explicit SourceLocation(

        #if defined __has_builtin
        #if __has_builtin(__builtin_FILE)
            const char *fileName = __builtin_FILE(),
        #else
            const char *fileName = "unimplemented",
        #endif
        #if __has_builtin(__builtin_FUNCTION)
            const char *functionName = __builtin_FUNCTION(),
        #else
            const char *functionName = "unimplemented",
        #endif
        #if __has_builtin(__builtin_LINE)
            const u32 lineNumber = __builtin_LINE(),
        #else
            const u32 lineNumber = 0,
        #endif
        #if __has_builtin(__builtin_COLUMN)
            const u32 columnOffset = __builtin_COLUMN()
        #else
            const u32 columnOffset = 0
        #endif
        #else

            const char *fileName = "unimplemented",
            const char *functionName = "unimplemented",
            const u32 lineNumber = 0, const u32 columnOffset = 0
        #endif

        ) : 
            fileName(fileName), 
            functionName(functionName), 
            lineNumber(lineNumber),
            columnOffset(columnOffset) {}

    /**
     * @brief format the location in multiple lines
     * 
     * @return std::string the formated location
     */
    std::string format_multiline(){
        return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
-------------------------
)="
            , fileName, lineNumber, columnOffset, functionName);
    }

    /**
     * @brief format the location in multiple lines with a given stacktrace
     * 
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    std::string format_multiline(std::string stacktrace){
        return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
stacktrace : 
{}
-------------------------
)="
            , fileName, lineNumber, columnOffset, functionName,stacktrace);
    }

    /**
     * @brief format the location in a one liner
     * 
     * @return std::string the formated location
     */
    std::string format_one_line(){
        return fmt::format("{}:{}:{}", fileName, lineNumber, columnOffset);
    }

    /**
     * @brief format the location in a one liner with the function name displayed
     * 
     * @return std::string the formated location
     */
    std::string format_one_line_func(){
        return fmt::format("{} ({}:{}:{})", functionName, fileName, lineNumber, columnOffset);
    }
};
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
 * @file logger.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <cstdarg>
#include <stdio.h>
#include <stdlib.h>
#include <string>

/**
 * @brief Logger class to pipe printf like outputs to files
 *
 */
class Logger {
    public:
    /**
     * @brief log file name
     */
    std::string log_file_name;

    /**
     * @brief internal pointer to FILE object
     */
    FILE *log_file;

    /**
     * @brief Construct a new Logger object
     *
     * @param filename filename which output would be piped to using log()
     */
    inline Logger(std::string filename) {
        log_file_name = filename;
        log_file      = fopen(log_file_name.c_str(), "w+");
    }

    /**
     * @brief Destroy the Logger object
     */
    inline ~Logger() { fclose(log_file); }

    /**
     * @brief printf like syntax to log to file
     *
     * @param aFormat
     * @param ...
     */
    inline void log(const char *aFormat, ...) {
        va_list argptr;
        va_start(argptr, aFormat);

        vfprintf(log_file, aFormat, argptr);

        va_end(argptr);
    }
};

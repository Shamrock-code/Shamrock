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
 * @file string.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "exception.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <array>
#include <fstream>
#include <vector>

namespace shambase {

    /**
     * @brief format a string using fmtlib style
     * Cheat sheet : https://hackingcpp.com/cpp/libs/fmt.html
     *
     * @tparam T
     * @param fmt the format string
     * @param args the arguments to format agains
     * @return std::string the formatted string
     */
    template<typename... T>
    inline std::string format(fmt::format_string<T...> fmt, T &&...args) {
        try {
            return fmt::format(fmt, args...);
        } catch (const std::exception &e) {
            throw make_except_with_loc<std::invalid_argument>(
                "format failed : " + std::string(e.what()));
        }
    }

    /**
     * @brief format a string using C printf style
     * https://cplusplus.com/reference/cstdio/printf/
     *
     * @tparam T
     * @param fmt the format string
     * @param args the arguments to format agains
     * @return std::string the formatted string
     */
    template<typename... T>
    inline std::string format_printf(std::string format, const T &...args) {
        try {
            return fmt::sprintf(format, args...);
        } catch (const std::exception &e) {

            throw make_except_with_loc<std::invalid_argument>(
                "format failed : " + std::string(e.what())
                + "\n fmt string : " + std::string(format));
        }
    }

    /**
     * @brief Format an array of elements into a string
     *
     * This function takes an iterator to the first element of the array,
     * the length of the array, the number of elements per column, and a format
     * string. It formats each element of the array using the format string and
     * concatenates the formatted elements into a single string.
     *
     * @tparam It Iterator type (deduced)
     * @tparam Tformat variadic template type (deduced)
     * @param iter iterator to the first element of the array
     * @param len length of the array
     * @param column_count number of elements per column
     * @param fmt format string
     * @return std::string the formatted string
     *
     * @throws std::invalid_argument if the format string is invalid
     */
    template<class It, typename... Tformat>
    inline std::string
    format_array(It &iter, u32 len, u32 column_count, fmt::format_string<Tformat...> fmt) {

        std::string accum;

        for (u32 i = 0; i < len; i++) {

            if (i % column_count == 0) {
                if (i == 0) {
                    accum += shambase::format("{:8} : ", i);
                } else {
                    accum += shambase::format("\n{:8} : ", i);
                }
            }

            accum += shambase::format(fmt, iter[i]);
        }

        return accum;
    }

    /**
     * @brief given a sizeof value return a readble string
     * Exemple : readable_sizeof(1024*1024*1024) -> "1.00 GB"
     *
     * @param size the size
     * @return std::string the formated string
     */
    inline std::string readable_sizeof(double size) {

        i32 i = 0;

        using namespace std::string_literals;
        const std::array units{"B"s, "kB"s, "MB"s, "GB"s, "TB"s, "PB"s, "EB"s, "ZB"s, "YB"s};

        if (size >= 0) {
            while (size > 1024) {
                size /= 1024;
                i++;
            }
        } else {
            i = 9;
        }

        if (i > 8) {
            return format_printf("%s", "err val");
        } else {
            return format_printf("%.2f %s", size, units[i]);
        }
    }

    /**
     * @brief dump a string to a file
     *
     * @param filename the filename
     * @param s the string to dump
     */
    inline void write_string_to_file(std::string filename, std::string s) {
        std::ofstream myfile(filename);
        myfile << s;
        myfile.close();
    }

    /**
     * @brief replace all occurence of a search string with another
     *
     * taken from https://en.cppreference.com/w/cpp/string/basic_string/replace
     *
     * @param inout the string to modify
     * @param what the search string
     * @param with the replace string
     */
    inline void replace_all(std::string &inout, std::string_view what, std::string_view with) {
        for (std::string::size_type pos{};
             inout.npos != (pos = inout.find(what.data(), pos, what.length()));
             pos += with.length()) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
    }

    /**
     * @brief Increase indentation of a string
     *
     * @param in the input string
     * @return std::string the output string
     */
    inline std::string increase_indent(std::string in, std::string delim = "\n    ") {
        std::string out = in;
        replace_all(out, "\n", delim);
        return "    " + out;
    }

    /**
     * @brief Truncate a string to a specified length, adding an ellipsis if necessary.
     *
     * @param s The string to truncate.
     * @param max_len The maximum length of the string.
     * @return std::string The truncated string.
     * @throws std::invalid_argument If the maximum length is less than 5.
     *
     * This function truncates a string to a specified length by taking the first `max_len - 5`
     * characters and appending an ellipsis if the original string is longer than `max_len`. If
     * `max_len` is less than 5, an `std::invalid_argument` exception is thrown.
     */
    inline std::string trunc_str(std::string s, u32 max_len) {

        if (max_len < 5)
            throw std::invalid_argument("max len should be above 4");

        if (s.size() > max_len) {
            return s.substr(0, max_len - 5) + " ...";
        } else {
            return s;
        }
    }

    /**
     * @brief Truncate a string to a specified length, adding an ellipsis at the start if necessary.
     *
     * @param s The input string.
     * @param max_len The maximum length of the string.
     * @return std::string The truncated string.
     * @throws std::invalid_argument If the maximum length is less than 5.
     *
     * This function truncates a string to a specified length by adding an ellipsis at the start of
     * the string, and taking the last `max_len - 4` characters if the original string is longer
     * than `max_len`. If `max_len` is less than 5, an `std::invalid_argument` exception is thrown.
     */
    inline std::string trunc_str_start(std::string s, u32 max_len) {
        if (max_len < 5)
            throw std::invalid_argument("max len should be above 4");

        if (s.size() > max_len) {
            return "... " + s.substr(s.size() - (max_len - 4), s.size());
        } else {
            return s;
        }
    }

    /**
     * @brief Check if a substring is present in a given string
     *
     * @param str The input string
     * @param what The substring to search for
     * @return true If the substring is found
     * @return false If the substring is not found
     */
    inline bool contain_substr(std::string str, std::string what) {
        return (str.find(what) != std::string::npos);
    }

    /**
     * @brief Shortens a string by removing the last specified number of characters
     *
     * @param str The input string
     * @param len The number of characters to be removed from the end of the string
     *
     * @throws std::invalid_argument If the length of the input string is less than the specified
     * length
     *
     * @return std::string The shortened string
     */
    inline std::string shorten_string(std::string str, u32 len) {
        if (len > str.size()) {
            throw make_except_with_loc<std::invalid_argument>(
                "the string is too short to be shorten"
                "\n args : "
                + format("{} : {} \n {} : {}", "str", str, "len", len));
        }
        return str.substr(0, str.size() - len);
    }

    /**
     * @brief Splits a string into a vector of substrings according to a delimiter
     *
     * @param s The string to be split
     * @param delimiter The string to be used as a delimiter for splitting
     * @return std::vector<std::string> The vector of substrings
     */
    inline std::vector<std::string> split_str(std::string s, std::string delimiter) {
        std::vector<std::string> ret;

        size_t pos = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            std::string substr = s.substr(0, pos);
            if (substr.size() > 0)
                ret.push_back(substr);
            s.erase(0, pos + delimiter.length());
        }
        if (s.size() > 0)
            ret.push_back(s);

        return ret;
    }

} // namespace shambase

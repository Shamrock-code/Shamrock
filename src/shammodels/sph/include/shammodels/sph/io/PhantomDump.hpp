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
 * @file PhantomDump.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/fortran_io.hpp"
#include "shambase/string.hpp"
#include <unordered_map>
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shammodels::sph {

    /**
     * @brief Phantom dump table header for a specific type
     *
     * This class is used to store the header of a Phantom dump file for one type.
     *
     * @tparam T type of the data
     */
    template<class T>
    struct PhantomDumpTableHeader {
        /**
         * @brief A vector of pairs containing the name and the value of each entry in
         * the table.
         */
        std::vector<std::pair<std::string, T>> entries;

        /**
         * @brief Reads the header from a Phantom dump file.
         *
         * @param phfile the file to read from
         * @return the header
         */
        static PhantomDumpTableHeader<T> from_file(shambase::FortranIOFile &phfile);

        /**
         * @brief Adds an entry to the header.
         *
         * @param s the name of the entry
         * @param val the value of the entry
         */
        void add(std::string s, T val) {
            s = shambase::format("{:16s}", s);
            entries.push_back({s, val});
        }

        /**
         * @brief Writes the header to a Phantom dump file.
         *
         * @param phfile the file to write to
         */
        void write(shambase::FortranIOFile &phfile);

        /**
         * @brief Fetches the value of a given entry from the header.
         *
         * @param s the name of the entry
         * @return the value of the entry
         */
        inline std::optional<T> fetch(std::string s) const {
            std::optional<T> ret = {};

            for (auto [key, val] : entries) {
                if (key == s) {
                    ret = val;
                }
            }

            return ret;
        }

        /**
         * @brief Fetches the values of all entries with a given name from the header into the
         * supplied vector.
         *
         * @param vec the vector to store the values in
         * @param s the name of the entries
         */
        template<class Tb>
        inline void fetch_multiple(std::vector<Tb> &vec, std::string s) {
            for (auto [key, val] : entries) {
                if (key == s) {
                    vec.push_back(val);
                }
            }
        }

        /**
         * @brief Prints the state of the header.
         */
        void print_state();

        template<class Tconv>
        inline void add_to_map(std::unordered_map<std::string, Tconv> &map) {
            for (auto [key, val] : entries) {
                map[key] = val;
            }
        }
    };

    /**
     * @brief A helper class to represent a single block of data in a Phantom dump.
     *
     * A single block of data in a Phantom dump consists of a 16-character tag, followed by a
     * variable number of values of a given type. This class represents such a block.
     *
     * @tparam T The type of the values in the block.
     */
    template<class T>
    struct PhantomDumpBlockArray {

        /**
         * @brief The tag of the block.
         */
        std::string tag;

        /**
         * @brief The values of the block.
         */
        std::vector<T> vals;

        /**
         * @brief Reads a block from a file
         *
         * @param phfile the file to read from
         * @param tot_count the total number of values to read
         * @return the block that was read
         */
        static PhantomDumpBlockArray from_file(shambase::FortranIOFile &phfile, i64 tot_count);

        /**
         * @brief Writes a block to a file
         *
         * @param phfile the file to write to
         * @param tot_count the total number of values to write
         */
        void write(shambase::FortranIOFile &phfile, i64 tot_count);

        /**
         * @brief Fills a vector with the values of a given field name
         *
         * @param field_name the name of the field to look for
         * @param vec the vector to fill
         */
        template<class Tb>
        void fill_vec(std::string field_name, std::vector<Tb> &vec) {
            if (tag == field_name) {
                for (T a : vals) {
                    vec.push_back(a);
                }
            }
        }

        /**
         * @brief Prints the state of the block
         */
        void print_state();
    };

    /**
     * @brief A class to represent a single block of data in a Phantom dump.
     *
     * @tparam T The type of the values in the block.
     */
    struct PhantomDumpBlock {
        /// The total number of values in the block.
        i64 tot_count;

        /// The type for Phantom's real type.
        using fort_real = f64;
        /// The type for Phantom's integer type.
        using fort_int = int;

        /// The blocks of values of type `fort_int`.
        std::vector<PhantomDumpBlockArray<fort_int>> blocks_fort_int;
        /// The blocks of values of type `i8`.
        std::vector<PhantomDumpBlockArray<i8>> blocks_i8;
        /// The blocks of values of type `i16`.
        std::vector<PhantomDumpBlockArray<i16>> blocks_i16;
        /// The blocks of values of type `i32`.
        std::vector<PhantomDumpBlockArray<i32>> blocks_i32;
        /// The blocks of values of type `i64`.
        std::vector<PhantomDumpBlockArray<i64>> blocks_i64;
        /// The blocks of values of type `fort_real`.
        std::vector<PhantomDumpBlockArray<fort_real>> blocks_fort_real;
        /// The blocks of values of type `f32`.
        std::vector<PhantomDumpBlockArray<f32>> blocks_f32;
        /// The blocks of values of type `f64`.
        std::vector<PhantomDumpBlockArray<f64>> blocks_f64;

        /**
         * @brief Gets the index of a block of type `fort_int` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_fort_int(std::string s);

        /**
         * @brief Gets the index of a block of type `i8` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_i8(std::string s);

        /**
         * @brief Gets the index of a block of type `i16` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_i16(std::string s);

        /**
         * @brief Gets the index of a block of type `i32` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_i32(std::string s);

        /**
         * @brief Gets the index of a block of type `i64` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_i64(std::string s);

        /**
         * @brief Gets the index of a block of type `fort_real` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_fort_real(std::string s);

        /**
         * @brief Gets the index of a block of type `f32` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_f32(std::string s);

        /**
         * @brief Gets the index of a block of type `f64` with the given name.
         *
         * @param s The name of the block to search for.
         * @return The index of the block in the vector, or `0` if no such block exists.
         */
        u64 get_ref_f64(std::string s);

        /**
         * @brief Prints the state of the block.
         */
        void print_state();

        /**
         * @brief Reads a block from a file
         *
         * @param phfile the file to read from
         * @param tot_count the total number of values to read
         * @param numarray the number of values of each type
         * @return the block that was read
         */
        static PhantomDumpBlock
        from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        /**
         * @brief Writes a block to a file
         *
         * @param phfile the file to write to
         * @param tot_count the total number of values to write
         * @param numarray the number of values of each type
         */
        void write(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        /**
         * @brief Fills a vector with the values of a given field name
         *
         * @param field_name the name of the field to look for
         * @param vec the vector to fill
         */
        template<class T>
        void fill_vec(std::string field_name, std::vector<T> &vec) {

            field_name = shambase::format("{:16s}", field_name);

            for (auto &tmp : blocks_fort_int) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_i8) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_i16) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_i32) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_i64) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_fort_real) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_f32) {
                tmp.fill_vec(field_name, vec);
            }
            for (auto &tmp : blocks_f64) {
                tmp.fill_vec(field_name, vec);
            }
        }
    };

    /**
     * @brief  Class representing a Phantom dump file.
     *
     * The class provides methods to read and write Phantom dump files, as well as to
     * retrieve data from the file.
     *
     * @todo add exemple of usage
     *
     */
    struct PhantomDump {

        /// Floating-point type used in the phantom dump format.
        using fort_real = f64;
        /// Integer type used in the phantom dump format.
        using fort_int = int;

        /// Magic number used in the phantom dump format.
        fort_int i1;
        /// Magic number used in the phantom dump format.
        fort_int i2;
        /// Magic number used in the phantom dump format.
        fort_int iversion;
        /// Magic number used in the phantom dump format.
        fort_int i3;
        /// Magic number used in the phantom dump format.
        fort_real r1;
        /// Magic number used in the phantom dump format.
        std::string fileid;

        /// Overrides the magic numbers used in the PhantomDump struct.
        void override_magic_number() {
            i1 = 60769;
            i2 = 60878;
            i3 = 690706;
            r1 = i2;
        }

        /**
         * @brief Checks if the magic numbers in the PhantomDump struct match the expected values.
         *
         * @throws std::runtime_error if any of the magic numbers do not match the expected values
         */
        void check_magic_numbers() {
            if (i1 != 60769) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (i2 != 60878) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (i3 != 690706) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (r1 != i2) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
        }

        /// Table header for integer data.
        PhantomDumpTableHeader<fort_int> table_header_fort_int;
        /// Table header for signed 8-bit integer data.
        PhantomDumpTableHeader<i8> table_header_i8;
        /// Table header for signed 16-bit integer data.
        PhantomDumpTableHeader<i16> table_header_i16;
        /// Table header for signed 32-bit integer data.
        PhantomDumpTableHeader<i32> table_header_i32;
        /// Table header for signed 64-bit integer data.
        PhantomDumpTableHeader<i64> table_header_i64;
        /// Table header for floating-point data.
        PhantomDumpTableHeader<fort_real> table_header_fort_real;
        /// Table header for 32-bit floating-point data.
        PhantomDumpTableHeader<f32> table_header_f32;
        /// Table header for 64-bit floating-point data.
        PhantomDumpTableHeader<f64> table_header_f64;
        /// List of blocks in the Phantom dump file.
        std::vector<PhantomDumpBlock> blocks;

        /**
         * @brief Generates a Phantom dump file from the current state of the object.
         *
         * @return a Phantom dump file
         */
        shambase::FortranIOFile gen_file();

        /**
         * @brief Reads a Phantom dump file and returns a PhantomDump object.
         *
         * @param phfile Phantom dump file to read
         * @return a PhantomDump object containing the data from the file
         */
        static PhantomDump from_file(shambase::FortranIOFile &phfile);

        /**
         * @brief Checks if a given string is present in any of the table headers.
         *
         * @param s the string to be searched in the table headers
         *
         * @return true if the string is found in any of the table headers, false otherwise
         */
        inline bool has_header_entry(std::string s) const {

            s = shambase::format("{:16s}", s);

            if (auto tmp = table_header_fort_int.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_i8.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_i16.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_i32.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_i64.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_fort_real.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_f32.fetch(s); tmp) {
                return true;
            }
            if (auto tmp = table_header_f64.fetch(s); tmp) {
                return true;
            }

            return false;
        }

        /**
         * @brief Retrieves a floating-point value from the table headers.
         *
         * @param s the string to be searched in the table headers
         *
         * @return the floating-point value associated with the given string
         *
         * @throws std::runtime_error if the entry cannot be found in the table headers
         */
        template<class T>
        inline T read_header_float(std::string s) const {

            s = shambase::format("{:16s}", s);

            if (auto tmp = table_header_fort_real.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_f32.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_f64.fetch(s); tmp) {
                return *tmp;
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "the entry cannot be found : " + s);

            return {};
        }

        /**
         * @brief Reads multiple float values from the table headers based on the given string.
         *
         * @param s the string to be searched in the table headers
         *
         * @return a vector of float values found in the table headers
         */
        template<class T>
        inline std::vector<T> read_header_floats(std::string s) {
            std::vector<T> vec{};

            s = shambase::format("{:16s}", s);

            table_header_fort_real.fetch_multiple(vec, s);
            table_header_f32.fetch_multiple(vec, s);
            table_header_f64.fetch_multiple(vec, s);

            return vec;
        }

        /**
         * @brief Retrieves an integer value from the table headers.
         *
         * @param s the string to be searched in the table headers
         *
         * @return the integer value associated with the given string
         *
         * @throws std::runtime_error if the entry cannot be found in the table headers
         */
        template<class T>
        inline T read_header_int(std::string s) const {

            s = shambase::format("{:16s}", s);

            if (auto tmp = table_header_fort_int.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_i8.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_i16.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_i32.fetch(s); tmp) {
                return *tmp;
            }
            if (auto tmp = table_header_i64.fetch(s); tmp) {
                return *tmp;
            }

            throw shambase::make_except_with_loc<std::runtime_error>("the entry cannot be found");

            return {};
        }

        /**
         * @brief Retrieves multiple integer values from the table headers based on the given
         * string.
         *
         * @param s the string to be searched in the table headers
         *
         * @return a vector of integer values found in the table headers
         */
        template<class T>
        inline std::vector<T> read_header_ints(std::string s) {
            std::vector<T> vec{};

            s = shambase::format("{:16s}", s);

            table_header_fort_int.fetch_multiple(vec, s);
            table_header_i8.fetch_multiple(vec, s);
            table_header_i16.fetch_multiple(vec, s);
            table_header_i32.fetch_multiple(vec, s);
            table_header_i64.fetch_multiple(vec, s);

            return vec;
        }

        /// Print current state of the data stored in the class
        void print_state();
    };

    /// Compare two phantom dumps and report offenses
    bool compare_phantom_dumps(PhantomDump &dump1, PhantomDump &dump2);

} // namespace shammodels::sph

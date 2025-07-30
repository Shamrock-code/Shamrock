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
 * @file fortran_io.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include <array>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace shambase {

    /**
     * @brief Class for reading and writing Fortran-style binary files
     */
    class FortranIOFile {

        /**
         * @brief Check that the next 4 bytes in the buffer are equal to `fortran_byte`
         *
         * @param[in] buffer the input buffer
         * @param[in] fortran_byte the expected value of the next 4 bytes
         *
         * @throw std::runtime_error if the next 4 bytes are not equal to `fortran_byte`
         */
        inline void check_fortran_4byte(std::basic_stringstream<byte> &buffer, i32 fortran_byte) {
            i32 new_check = 0;

            shambase::stream_read(buffer, new_check);

            if (new_check != fortran_byte) {
                throw shambase::make_except_with_loc<std::runtime_error>("fortran 4 bytes invalid");
            }

            fortran_byte = new_check;
        }

        /**
         * @brief Read the next 4 bytes from the buffer.
         *
         * In Fortran every write call produce 4 bte specifying the length of the write followed by
         * the same 4 bytes. We can check them here to verify that we are reading the file
         * correctly.
         *
         * @param[in] buffer the input buffer
         *
         * @return the value of the next 4 bytes
         */
        inline i32 read_fortran_4byte(std::basic_stringstream<byte> &buffer) {
            i32 check;
            shambase::stream_read(buffer, check);
            return check;
        }

        /// Data stream of the file
        std::basic_stringstream<byte> data;

        /// Lenght of the data stream
        u64 length;

        /**
         * @brief Write a value to the buffer
         *
         * @tparam T The type of the value to write
         * @param arg The value to write
         */
        template<class T>
        inline void _write(T arg) {
            stream_write(data, arg);
        }

        /**
         * @brief Read a value from the buffer
         *
         * @tparam T The type of the value to read
         * @param arg The reference to the variable to store the read value
         */
        template<class T>
        inline void _read(T &arg) {
            stream_read(data, arg);
        }

        /**
         * @brief Read an array of values from the buffer
         *
         * @tparam T The type of the values in the array
         * @tparam N The size of the array
         * @param vec The reference to the array to store the read values
         */
        template<class T, int N>
        inline void _read(std::array<T, N> &vec) {
            for (u32 i = 0; i < N; i++) {
                stream_read(data, vec[i]);
            }
        }

        /**
         * @brief Write an array of values to the buffer
         *
         * @tparam T The type of the values in the array
         * @tparam N The size of the array
         * @param vec The reference to the array to write
         */
        template<class T, int N>
        inline void _write(std::array<T, N> &vec) {
            for (u32 i = 0; i < N; i++) {
                stream_write(data, vec[i]);
            }
        }

        public:
        /// Fortran real type
        using fort_real = f64;

        /// Fortran int type
        using fort_int = int;

        /**
         * @brief Construct a new FortranIOFile object
         *
         * @param[in] data_in The input buffer to be used for reading and writing
         * @param[in] length The length of the input buffer
         */
        explicit FortranIOFile(std::basic_stringstream<byte> &&data_in, u64 length)
            : data(std::forward<std::basic_stringstream<byte>>(data_in)), length(length) {

            // Set the internal buffer to the beginning of the buffer
            data.seekg(0);
        }

        FortranIOFile() = default;

        /**
         * @brief Get a reference to the internal buffer
         *
         * @return std::basic_stringstream<byte>& A reference to the internal buffer
         */
        inline std::basic_stringstream<byte> &get_internal_buf() {
            // Return a reference to the internal buffer
            return data;
        }

        /**
         * @brief Write a list of arguments to the internal buffer
         *
         * This function writes a list of arguments to the internal buffer using the
         * Fortran-like serialization format. The arguments are serialized in the following
         * way: first the size of all arguments in bytes, then the arguments, and finally
         * again the size of all arguments in bytes. This format is used by the Fortran
         * standard for I/O.
         *
         * @tparam Args the types of the arguments to be written
         * @param[in] args the arguments to be written
         */
        template<class... Args>
        inline void write(Args &...args) {
            i32 linebytecount = ((sizeof(args)) + ...);
            stream_write(data, linebytecount);
            ((_write(args)), ...);
            stream_write(data, linebytecount);
        }

        /**
         * @brief Read a list of arguments from the internal buffer
         *
         * This function reads a list of arguments from the internal buffer using the
         * Fortran-like serialization format. The arguments are serialized in the following
         * way: first the size of all arguments in bytes, then the arguments, and finally
         * again the size of all arguments in bytes.
         *
         * @tparam Args the types of the arguments to be read
         * @param[out] args the arguments to be read
         */
        template<class... Args>
        inline void read(Args &...args) {
            u64 linebytecount = ((sizeof(args)) + ...);
            i32 check         = read_fortran_4byte(data);
            if (check != linebytecount) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }
            ((_read(args)), ...);
            check_fortran_4byte(data, check);
        }

        /**
         * @brief Read a fixed-length string from the buffer
         *
         * This function reads a fixed-length string from the buffer using the
         * Fortran-like serialization format. The string is serialized in the
         * following way: first the size of the string in bytes, then the
         * string itself, and finally again the size of the string in bytes.
         *
         * @param[out] s the string to be read
         * @param len the length of the string to be read
         */
        inline void read_fixed_string(std::string &s, u32 len) {
            s.resize(len);
            i32 check = read_fortran_4byte(data);
            if (check != len) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }
            data.read(reinterpret_cast<byte *>(s.data()), len * sizeof(char));
            check_fortran_4byte(data, check);
        }

        /**
         * @brief Write a fixed-length string to the buffer
         *
         * This function writes a fixed-length string to the buffer using the
         * Fortran-like serialization format. The string is serialized in the
         * following way: first the size of the string in bytes, then the
         * string itself, and finally again the size of the string in bytes.
         *
         * @param[out] s the string to be written
         * @param[in] len the length of the string to be written
         */
        inline void write_fixed_string(std::string &s, u32 len) {
            stream_write(data, len);
            data.write(reinterpret_cast<byte *>(s.data()), len * sizeof(char));
            stream_write(data, len);
        }

        /**
         * @brief Read a fixed-length string array from the buffer
         *
         * This function reads a fixed-length string array from the buffer using
         * the Fortran-like serialization format. The array is serialized in the
         * following way: first the size of the array in bytes, then the array
         * itself, and finally again the size of the array in bytes.
         *
         * @param[out] svec the output array of strings
         * @param[in] strlen the length of each string in the array
         * @param[in] str_count the number of strings in the array
         */
        inline void read_string_array(std::vector<std::string> &svec, u32 strlen, u32 str_count) {

            u64 totlen = strlen * str_count;
            i32 check  = read_fortran_4byte(data);
            if (check != totlen) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }

            svec.resize(str_count);

            for (u32 i = 0; i < str_count; i++) {
                svec[i].resize(strlen); // Resize the string to the correct size
                data.read(reinterpret_cast<byte *>(svec[i].data()), strlen * sizeof(char));
            }

            check_fortran_4byte(data, check);
        }

        /**
         * @brief Write a fixed-length string array to the buffer
         *
         * This function writes a fixed-length string array to the buffer using
         * the Fortran-like serialization format. The array is serialized in the
         * following way: first the size of the array in bytes, then the array
         * itself, and finally again the size of the array in bytes.
         *
         * @param[in] svec the array of strings to write
         * @param[in] strlen the length of each string in the array
         * @param[in] str_count the number of strings in the array
         */
        inline void write_string_array(std::vector<std::string> &svec, u32 strlen, u32 str_count) {

            i32 totlen = strlen * str_count; // Total length of the array

            stream_write(data, totlen); // Write the total length as a 4-byte integer

            for (u32 i = 0; i < str_count; i++) {
                // Write each string to the buffer
                data.write(reinterpret_cast<byte *>(svec[i].data()), strlen * sizeof(char));
            }

            stream_write(data, totlen); // Write the total length again as a 4-byte integer
        }

        /**
         * @brief Read an array of values from the buffer
         *
         * This function reads an array of values of type T from the buffer using
         * the Fortran-like serialization format. The array is serialized in the
         * following way: first the size of the array in bytes, then the array
         * itself, and finally again the size of the array in bytes.
         *
         * @tparam T The type in use
         * @param[out] vec the output array
         * @param[in] val_count the number of values in the array
         */
        template<class T>
        inline void read_val_array(std::vector<T> &vec, u32 val_count) {

            u64 totlen = sizeof(T) * val_count;    // Total length of the array
            i32 check  = read_fortran_4byte(data); // Read the total length

            if (check != totlen) {                  // Make sure the byte count matches
                throw_with_loc<std::runtime_error>( // Throw an exception if not
                    "the byte count is not correct");
            }

            vec.resize(val_count); // Resize the output array to the correct size

            for (u32 i = 0; i < val_count; i++) { // Read each value from the buffer
                stream_read(data, vec[i]);
            }

            check_fortran_4byte(data, check); // Check the byte count again
        }

        /**
         * @brief Write an array of values to the buffer
         *
         * This function writes an array of values of type T to the buffer
         * using the Fortran-like serialization format. The array is
         * serialized in the following way: first the size of the array in
         * bytes, then the array itself, and finally again the size of the
         * array in bytes.
         *
         * @param vec The input array to be written
         * @param val_count The number of values in the array
         *
         * @throw std::invalid_argument if val_count is larger than vec.size()
         */
        template<class T>
        inline void write_val_array(std::vector<T> &vec, u32 val_count) {
            if (val_count > vec.size()) {
                throw make_except_with_loc<std::invalid_argument>(
                    "val count is higher than vec size");
            }
            i32 totlen = sizeof(T) * val_count;
            stream_write(data, totlen); // Write the total length of the array

            for (u32 i = 0; i < val_count; i++) { // Write each value in the array
                stream_write(data, vec[i]);
            }

            stream_write(data, totlen); // Write the total length again
        }

        /**
         * @brief Check if the end of the file has been reached
         *
         * This function returns true if the end of the file has been reached,
         * i.e. if all the data in the file has been read.
         *
         * @return true if the end of the file has been reached
         */
        inline bool finished_read() { return length == data.tellg(); }

        /**
         * @brief Write the Fortran formatted file to disk.
         *
         * @param fname Filename to write to
         *
         * @throws runtime_error if the file could not be opened for writing
         */
        inline void write_to_file(std::string fname) {
            std::ofstream out_f(fname, std::ios::binary);

            if (out_f) {
                // Copy the contents of the internal streambuf to the file
                out_f << data.rdbuf();

                // Close the file
                out_f.close();
            } else {
                // Throw an exception if the file could not be opened for writing
                throw_unimplemented(
                    "unimplemented case : could not open file " + fname + " for writing");
            }
        }
    };

    /**
     * @brief Load a Fortran formatted file from disk.
     *
     * @param fname Filename of the file to load
     *
     * @return The loaded file
     *
     * @throws runtime_error if the file is not found
     */
    inline FortranIOFile load_fortran_file(std::string fname) {
        std::ifstream in_f(fname, std::ios::binary);

        std::basic_stringstream<byte> buffer;
        if (in_f) {
            buffer << in_f.rdbuf();
            in_f.close();
        } else {
            throw_unimplemented("unimplemented case : file not found");
        }

        return FortranIOFile(std::move(buffer), buffer.tellp());
    }

} // namespace shambase

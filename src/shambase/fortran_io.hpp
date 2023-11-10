// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file fortran_io.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include "shamsys/legacy/log.hpp"
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace shambase {

    class FortranIOFile {

        inline void check_fortran_4byte(std::basic_stringstream<byte> &buffer, i32 fortran_byte) {
            i32 new_check = 0;

            shambase::stream_read(buffer, new_check);

            if (new_check != fortran_byte) {
                throw shambase::throw_with_loc<std::runtime_error>("fortran 4 bytes invalid");
            }

            fortran_byte = new_check;
        }

        inline i32 read_fortran_4byte(std::basic_stringstream<byte> &buffer) {
            i32 check;
            shambase::stream_read(buffer, check);
            return check;
        }

        std::basic_stringstream<byte> data;
        u64 lenght;

        template<class T>
        inline void _write(T arg) {
            stream_write(data, arg);
        }

        template<class T>
        inline void _read(T &arg) {
            stream_read(data, arg);
        }

        template<class T, int N>
        inline void _read(std::array<T, N> &vec) {
            for (u32 i = 0; i < N; i++) {
                stream_read(data, vec[i]);
            }
        }

        template<class T, int N>
        inline void _write(std::array<T, N> &vec) {
            for (u32 i = 0; i < N; i++) {
                stream_write(data, vec[i]);
            }
        }

        public:
        using fort_real = f64;
        using fort_int  = int;

        explicit FortranIOFile(std::basic_stringstream<byte> &&data_in, u64 lenght)
            : data(std::forward<std::basic_stringstream<byte>>(data_in)), lenght(lenght) {

            data.seekg(0);
        }

        FortranIOFile() = default;

        inline std::basic_stringstream<byte> &get_internal_buf() { return data; }

        template<class... Args>
        inline void write(Args &...args) {
            i32 linebytecount = ((sizeof(args)) + ...);
            stream_write(data, linebytecount);
            ((_write(args)), ...);
            stream_write(data, linebytecount);
        }

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

        inline void read_fixed_string(std::string &s, u32 len) {
            s.resize(len);
            i32 check = read_fortran_4byte(data);
            if (check != len) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }
            data.read(reinterpret_cast<byte *>(s.data()), len * sizeof(char));
            check_fortran_4byte(data, check);
        }

        inline void write_fixed_string(std::string &s, u32 len) {
            stream_write(data, len);
            data.write(reinterpret_cast<byte *>(s.data()), len * sizeof(char));
            stream_write(data, len);
        }

        inline void read_string_array(std::vector<std::string> &svec, u32 strlen, u32 str_count) {

            u64 totlen = strlen * str_count;
            i32 check  = read_fortran_4byte(data);
            if (check != totlen) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }

            svec.resize(str_count);

            for (u32 i = 0; i < str_count; i++) {
                svec[i].resize(strlen);
                data.read(reinterpret_cast<byte *>(svec[i].data()), strlen * sizeof(char));
            }

            check_fortran_4byte(data, check);
        }

        inline void write_string_array(std::vector<std::string> &svec, u32 strlen, u32 str_count) {
            
            i32 totlen = strlen * str_count;

            stream_write(data, totlen);

            for (u32 i = 0; i < str_count; i++) {
                data.write(reinterpret_cast<byte *>(svec[i].data()), strlen * sizeof(char));
            }

            stream_write(data, totlen);
        }

        template<class T>
        inline void read_val_array(std::vector<T> &vec, u32 val_count) {

            u64 totlen = sizeof(T) * val_count;
            i32 check  = read_fortran_4byte(data);
            if (check != totlen) {
                throw_with_loc<std::runtime_error>("the byte count is not correct");
            }

            vec.resize(val_count);

            for (u32 i = 0; i < val_count; i++) {
                stream_read(data, vec[i]);
            }

            check_fortran_4byte(data, check);
        }

        template<class T>
        inline void write_val_array(std::vector<T> &vec, u32 val_count) {

            i32 totlen = sizeof(T) * val_count;
            stream_write(data, totlen);

            for (u32 i = 0; i < val_count; i++) {
                stream_write(data, vec[i]);
            }

            stream_write(data, totlen);
        }

        inline bool finished_read() { return lenght == data.tellg(); }

        inline void write_to_file(std::string fname) {
            std::ofstream out_f(fname, std::ios::binary);

            if (out_f) {
                out_f << data.rdbuf();

                out_f.close();
            } else {
                shambase::throw_unimplemented();
            }
        }
    };

    inline FortranIOFile load_fortran_file(std::string fname) {
        std::ifstream in_f(fname, std::ios::binary);

        std::basic_stringstream<byte> buffer;
        if (in_f) {
            buffer << in_f.rdbuf();
            in_f.close();
        } else {
            shambase::throw_unimplemented();
        }

        return FortranIOFile(std::move(buffer), buffer.tellp());
    }

} // namespace shambase
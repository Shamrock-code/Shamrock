// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PhantomDump.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include "shambase/fortran_io.hpp"
#include "shambase/string.hpp"
#include "shammodels/EOSConfig.hpp"
#include "shammodels/sph/AVConfig.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace shammodels::sph {

    template<class T>
    struct PhantomDumpTableHeader {
        std::vector<std::pair<std::string, T>> entries;

        static PhantomDumpTableHeader<T> from_file(shambase::FortranIOFile &phfile);

        void write(shambase::FortranIOFile &phfile);

        inline std::optional<T> fetch(std::string s){
            std::optional<T> ret = {};

            for(auto [key,val] : entries){
                if(key == s){
                    ret = val;
                }
            }

            return ret;
        }
    };

    template<class T>
    struct PhantomDumpBlockArray {

        std::string tag;
        std::vector<T> vals;

        static PhantomDumpBlockArray from_file(shambase::FortranIOFile &phfile, i64 tot_count);

        void write(shambase::FortranIOFile &phfile, i64 tot_count);


        template<class Tb>
        void fill_vec(std::string field_name, std::vector<Tb> & vec){
            if(tag == field_name){
                for(T a : vals){
                    vec.push_back(a);
                }
            }
        }
    };

    struct PhantomDumpBlock {
        i64 tot_count;

        using fort_real = f64;
        using fort_int  = int;

        std::vector<PhantomDumpBlockArray<fort_int>> table_header_fort_int;
        std::vector<PhantomDumpBlockArray<i8>> table_header_i8;
        std::vector<PhantomDumpBlockArray<i16>> table_header_i16;
        std::vector<PhantomDumpBlockArray<i32>> table_header_i32;
        std::vector<PhantomDumpBlockArray<i64>> table_header_i64;
        std::vector<PhantomDumpBlockArray<fort_real>> table_header_fort_real;
        std::vector<PhantomDumpBlockArray<f32>> table_header_f32;
        std::vector<PhantomDumpBlockArray<f64>> table_header_f64;

        static PhantomDumpBlock
        from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        void write(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);
    
        template<class T>
        void fill_vec(std::string field_name, std::vector<T> & vec){

            field_name = shambase::format("{:16s}",field_name);

            for(auto & tmp : table_header_fort_int){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_i8){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_i16){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_i32){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_i64){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_fort_real){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_f32){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : table_header_f64){
                tmp.fill_vec(field_name, vec);
            }
        }


    };

    struct PhantomDump {

        using fort_real = f64;
        using fort_int  = int;

        fort_int i1, i2, iversion, i3;
        fort_real r1;
        std::string fileid;

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

        PhantomDumpTableHeader<fort_int> table_header_fort_int;
        PhantomDumpTableHeader<i8> table_header_i8;
        PhantomDumpTableHeader<i16> table_header_i16;
        PhantomDumpTableHeader<i32> table_header_i32;
        PhantomDumpTableHeader<i64> table_header_i64;
        PhantomDumpTableHeader<fort_real> table_header_fort_real;
        PhantomDumpTableHeader<f32> table_header_f32;
        PhantomDumpTableHeader<f64> table_header_f64;

        std::vector<PhantomDumpBlock> blocks;

        shambase::FortranIOFile gen_file();

        static PhantomDump from_file(shambase::FortranIOFile &phfile);

        template<class T>
        inline T read_header_float(std::string s){

            s = shambase::format("{:16s}",s);

            if(auto tmp = table_header_fort_real.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_f32.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_f64.fetch(s); tmp){
                return *tmp;
            }

            throw shambase::throw_with_loc<std::runtime_error>("the entry cannot be found");

            return {};
        }

        template<class T>
        inline T read_header_int(std::string s){

            s = shambase::format("{:16s}",s);

            if(auto tmp = table_header_fort_int.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_i8.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_i16.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_i32.fetch(s); tmp){
                return *tmp;
            }
            if(auto tmp = table_header_i64.fetch(s); tmp){
                return *tmp;
            }

            throw shambase::throw_with_loc<std::runtime_error>("the entry cannot be found");

            return {};
        }

    };

    template<class Tvec>
    EOSConfig<Tvec> get_shamrock_eosconfig(PhantomDump & phdump);

    template<class Tvec>
    AVConfig<Tvec> get_shamrock_avconfig(PhantomDump & phdump);


} // namespace shammodels::sph
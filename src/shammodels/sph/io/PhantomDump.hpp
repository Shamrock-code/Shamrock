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
#include "shammodels/sph/config/AVConfig.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamunits/UnitSystem.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <functional>
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

        void add(std::string s, T val){
            s = shambase::format("{:16s}", s);
            entries.push_back({s,val});
        }

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

        template<class Tb>
        inline void fetch_multiple(std::vector<Tb> &vec, std::string s){
            for(auto [key,val] : entries){
                if(key == s){
                    vec.push_back(val);
                }
            }
        }

        void print_state();

        
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

        void print_state();
    };

    struct PhantomDumpBlock {
        i64 tot_count;

        using fort_real = f64;
        using fort_int  = int;

        std::vector<PhantomDumpBlockArray<fort_int>> blocks_fort_int;
        std::vector<PhantomDumpBlockArray<i8>> blocks_i8;
        std::vector<PhantomDumpBlockArray<i16>> blocks_i16;
        std::vector<PhantomDumpBlockArray<i32>> blocks_i32;
        std::vector<PhantomDumpBlockArray<i64>> blocks_i64;
        std::vector<PhantomDumpBlockArray<fort_real>> blocks_fort_real;
        std::vector<PhantomDumpBlockArray<f32>> blocks_f32;
        std::vector<PhantomDumpBlockArray<f64>> blocks_f64;

        u64 get_ref_fort_int(std::string s);
        u64 get_ref_i8(std::string s);
        u64 get_ref_i16(std::string s);
        u64 get_ref_i32(std::string s);
        u64 get_ref_i64(std::string s);
        u64 get_ref_fort_real(std::string s);
        u64 get_ref_f32(std::string s);
        u64 get_ref_f64(std::string s);


        void print_state();

        static PhantomDumpBlock
        from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        void write(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);
    
        template<class T>
        void fill_vec(std::string field_name, std::vector<T> & vec){

            field_name = shambase::format("{:16s}",field_name);

            for(auto & tmp : blocks_fort_int){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_i8){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_i16){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_i32){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_i64){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_fort_real){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_f32){
                tmp.fill_vec(field_name, vec);
            }
            for(auto & tmp : blocks_f64){
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

        void override_magic_number(){
            i1 = 60769;
            i2 = 60878;
            i3 = 690706;
            r1 = i2;
        }

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

        inline bool has_header_entry(std::string s){

            s = shambase::format("{:16s}",s);

            if(auto tmp = table_header_fort_int.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_i8.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_i16.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_i32.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_i64.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_fort_real.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_f32.fetch(s); tmp){
                return true;
            }
            if(auto tmp = table_header_f64.fetch(s); tmp){
                return true;
            }

            return false;
        }

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

            throw shambase::make_except_with_loc<std::runtime_error>("the entry cannot be found : "+s);

            return {};
        }

        template<class T>
        inline std::vector<T> read_header_floats(std::string s){
            std::vector<T> vec {};

            s = shambase::format("{:16s}",s);

            table_header_fort_real.fetch_multiple(vec, s);
            table_header_f32.fetch_multiple(vec, s);
            table_header_f64.fetch_multiple(vec, s);

            return vec;
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

            throw shambase::make_except_with_loc<std::runtime_error>("the entry cannot be found");

            return {};
        }

        template<class T>
        inline std::vector<T> read_header_ints(std::string s){
            std::vector<T> vec {};

            s = shambase::format("{:16s}",s);

            table_header_fort_int.fetch_multiple(vec, s);
            table_header_i8.fetch_multiple(vec, s);
            table_header_i16.fetch_multiple(vec, s);
            table_header_i32.fetch_multiple(vec, s);
            table_header_i64.fetch_multiple(vec, s);

            return vec;
        }

        void print_state();

    };

    template<class Tvec>
    EOSConfig<Tvec> get_shamrock_eosconfig(PhantomDump & phdump, bool bypass_error);

    template<class Tvec>
    AVConfig<Tvec> get_shamrock_avconfig(PhantomDump & phdump);

    /**
     * @brief Get the shamrock units object
     * \todo load also magfd
     * @tparam Tscal 
     * @param phdump 
     * @return shamunits::UnitSystem<Tscal> 
     */
    template<class Tscal> 
    shamunits::UnitSystem<Tscal> get_shamrock_units(PhantomDump & phdump);


} // namespace shammodels::sph
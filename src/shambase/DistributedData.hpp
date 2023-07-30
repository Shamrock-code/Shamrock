// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamsys/legacy/log.hpp"
#include <map>
#include <utility>

namespace shambase {

    /**
     * @brief Describe an object distributed accros patches (id = u64)
     *
     * @tparam T
     */
    template<class T>
    class DistributedData {

        std::map<u64, T> data;

        using iterator = typename std::map<u64, T>::iterator;

        public:
        inline std::map<u64, T> &get_native() { return data; }

        inline iterator add_obj(u64 id, T &&obj) {

            std::pair<iterator, bool> ret = data.emplace(id, std::forward<T>(obj));

            if (!ret.second) {
                throw throw_with_loc<std::runtime_error>("the key already exist");
            }

            return ret.first;
        }

        inline void erase(u64 id) { data.erase(id); }

        inline void for_each(std::function<void(u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }
        inline auto find(u64 id) { return data.find(id); }

        inline auto not_found() { return data.end(); }

        inline T &get(u64 id) { return data.at(id); }

        inline bool has_key(u64 id) { return (data.find(id) != data.end()); }

        inline u64 get_element_count() { return data.size(); }

        inline bool is_empty(){
            return data.empty();
        }

        template<typename... Tf>
        inline void print_data(fmt::format_string<Tf...> fmt){
            for_each([&](u64 id_patch, T & ref){
                logger::raw_ln(id_patch ,"->" ,shambase::format(fmt, ref));
            });
        }

        template<class Tmap>
        inline DistributedData<Tmap> map(std::function<Tmap(u64, T&)> map_func){
            DistributedData<Tmap> ret;
            for_each([&](u64 id, T& ref){
                ret.add_obj(id, map_func(id,ref));
            });
            return ret;
        }

        inline void reset(){
            data.clear();
        }
    };

    /**
     * @brief Describe an object common to two patches, typically interface (sender,receiver)
     *
     * @tparam T
     */
    template<class T>
    class DistributedDataShared {

        std::multimap<std::pair<u64, u64>, T> data;

        using iterator = typename std::multimap<std::pair<u64, u64>, T>::iterator;

        public:
        inline std::multimap<std::pair<u64, u64>, T> &get_native() { return data; }

        inline iterator add_obj(u64 left_id, u64 right_id, T &&obj) {
            std::pair<u64, u64> tmp = {left_id, right_id};
            return data.emplace(std::move(tmp), std::forward<T>(obj));
        }

        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        inline void tranfer_all(std::function<bool(u64, u64)> cd, DistributedDataShared & other){

            std::vector<std::pair<u64, u64>> occurences;

            // whoa i forgot the & here and triggered the copy constructor of every patch
            // like do not forget it or it will be a disaster waiting to come
            // i did throw up a 64 GPUs run because of that
            for(auto & [k,v] : data){
                if(cd(k.first,k.second)){
                    occurences.push_back(k);
                }
            }

            for(auto p : occurences){
                auto ext = data.extract(p);
                other.data.insert(std::move(ext));
            }

        }

        inline bool has_key(u64 left_id, u64 right_id) {
            return (data.find({left_id, right_id}) != data.end());
        }

        inline u64 get_element_count() { return data.size(); }

        template<class Tmap>
        inline DistributedDataShared<Tmap> map(std::function<Tmap(u64, u64, T&)> map_func){
            DistributedDataShared<Tmap> ret;
            for_each([&](u64 left,u64 right, T& ref){
                ret.add_obj(left, right, map_func(left,right,ref));
            });
            return ret;
        }
        
        inline void reset(){
            data.clear();
        }

        inline bool is_empty(){
            return data.empty();
        }
    };
} // namespace shambase

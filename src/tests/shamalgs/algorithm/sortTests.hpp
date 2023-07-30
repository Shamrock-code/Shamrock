// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/algorithm.hpp"
#include "shamalgs/memory.hpp"
#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/random.hpp"
#include <numeric>

template<class Tkey, class Tval>
struct TestSortByKey {

    using vFunctionCall = void (*)(sycl::queue &, sycl::buffer<Tkey> &, sycl::buffer<Tval> &, u32 );

    vFunctionCall fct;

    explicit TestSortByKey(vFunctionCall arg) : fct(arg){};

    void check() {

        sycl::queue & q = shamsys::instance::get_compute_queue();

        u32 len = 1U << 8U;
        
        sycl::buffer<u32> buf_key = shamalgs::random::mock_buffer<u32>(0x111, len, 0,1U << 12U);
        std::vector<u32> key_before_sort = shamalgs::memory::buf_to_vec(buf_key, len);

        sycl::buffer<u32> buf_vals = shamalgs::algorithm::gen_buffer_index(q, len);

        fct(q, buf_key, buf_vals, len);

        std::vector<u32> sorted_keys = shamalgs::memory::buf_to_vec(buf_key, len);
        std::vector<u32> sorted_vals = shamalgs::memory::buf_to_vec(buf_vals, len);

        bool sort_ok = std::is_sorted(sorted_keys.begin(), sorted_keys.end());

        bool check_map = true;
        for (u32 i = 0 ; i < len; i++) {
            check_map = check_map && ( sorted_keys[i] == key_before_sort[sorted_vals[i]] );
        }

        shamtest::asserts().assert_bool("is sorted", sort_ok);
        shamtest::asserts().assert_bool("values permutation ok", check_map);

    }

    f64 benchmark_one(u32 len){

        sycl::queue & q = shamsys::instance::get_compute_queue();
        
        sycl::buffer<u32> buf_key = shamalgs::random::mock_buffer<u32>(0x111, len, 0,1U << 31U);
        sycl::buffer<u32> buf_vals = shamalgs::algorithm::gen_buffer_index(q, len);


        shamalgs::memory::move_buffer_on_queue(q, buf_key);
        shamalgs::memory::move_buffer_on_queue(q, buf_vals);

        q.wait();

        shambase::Timer t;
        t.start();
        fct(q, buf_key, buf_vals, len);
        q.wait();t.end();

        return (t.nanosec*1e-9);
    }

    f64 bench_one_avg(u32 len){
        f64 sum = 0;

        f64 cnt = 1;

        if(len < 2e6){
            cnt = 4;
        }else if(len < 1e5){
            cnt = 10;
        }else if(len < 1e4){
            cnt = 100;
        }


        for(u32 i = 0; i < cnt; i++){
            sum += benchmark_one(len);
        }

        return sum / cnt;
    }

    struct BenchRes{
        std::vector<f64> sizes;
        std::vector<f64> times;
    };

    BenchRes benchmark(){
        BenchRes ret;
        
        logger::info_ln("TestSortByKey","testing :",__PRETTY_FUNCTION__);

        constexpr u32 lim_bench = 1.5e8;
        for(f64 i = 16; i < lim_bench; i*=2){
            ret.sizes.push_back(i);
        }




        for(const f64 & sz : ret.sizes){
            logger::debug_ln("ShamrockTest","N=",sz);
            ret.times.push_back(bench_one_avg(sz));
        }

        return ret;
    }
};


struct TestStreamCompact {

    using vFunctionCall = std::tuple<sycl::buffer<u32>, u32> (*)(sycl::queue&, sycl::buffer<u32> &, u32 );

    vFunctionCall fct;

    explicit TestStreamCompact(vFunctionCall arg) : fct(arg){};

    void check() {
        std::vector<u32> data {1,0,0,1,0,1,1,0,1,0,1,0,1};

        u32 len = data.size();

        auto buf = shamalgs::memory::vec_to_buf(data);

        auto [res, res_len] = fct(shamsys::instance::get_compute_queue(), buf, len);

        auto res_check = shamalgs::memory::buf_to_vec(res, res_len);

        //make check
        std::vector<u32> idxs ;
        {
            for(u32 idx = 0; idx < len; idx ++){
                if(data[idx]){
                    idxs.push_back(idx);
                }
            }
        }

        shamtest::asserts().assert_equal("same lenght", res_len, u32(idxs.size()));

        for(u32 idx = 0; idx < res_len; idx ++){
            shamtest::asserts().assert_equal("sid_check", res_check[idx], idxs[idx]);
        }
    }
};


template<class T>
struct TestIndexRemap{
using vFunctionCall = sycl::buffer<u32> (*)(sycl::queue &, sycl::buffer<T> &, sycl::buffer<u32> &, u32 );

    vFunctionCall fct;

    explicit TestIndexRemap(vFunctionCall arg) : fct(arg){};

    void check() {

        sycl::queue & q = shamsys::instance::get_compute_queue();

        u32 len = 1U << 5U;
        
        std::unique_ptr<sycl::buffer<u32>> buf_key  = std::make_unique<sycl::buffer<u32>>(
            shamalgs::random::mock_buffer<u32>(0x111, len, 0,1U << 7U)
        );

        std::unique_ptr<sycl::buffer<u32>> buf_key_dup  = std::make_unique<sycl::buffer<u32>>(
            shamalgs::random::mock_buffer<u32>(0x111, len, 0,1U << 7U)
        );

        sycl::buffer<u32> buf_index_map = shamalgs::algorithm::gen_buffer_index(q, len);
        shamalgs::algorithm::sort_by_key(q, *buf_key, buf_index_map, len);


        sycl::buffer<u32> remaped_key = fct(q, *buf_key_dup, buf_index_map, len);

        std::vector<u32> sorted_keys = shamalgs::memory::buf_to_vec(*buf_key, len);
        std::vector<u32> remaped_keys = shamalgs::memory::buf_to_vec(remaped_key, len);

        bool match = true;
        for (u32 i = 0 ; i < len; i++) {
            match = match && ( sorted_keys[i] == remaped_keys[i] );
        }

        shamtest::asserts().assert_bool("permutation is corect", match);

    }
};
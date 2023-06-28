// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <random>
#include "shamalgs/reduction/reduction.hpp"
#include "shamalgs/reduction/details/sycl2020reduction.hpp"
#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/random/random.hpp"
#include "shamalgs/reduction/details/groupReduction.hpp"

using namespace shamalgs::random;

template<class T,class Fct> void unit_test_reduc(std::string name, Fct && red_fct){
    std::vector<T> vals;

    constexpr u32 size_test = 1e6;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 1);


    for(u32 i = 0; i < size_test; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf {vals.data(),vals.size()};

        sycl_ret = red_fct(shamsys::instance::get_compute_queue(), buf, 0, size_test);

    }

    {
        check_val = T{0};
        for(auto & f : vals){
            check_val += f;
        }
    }

    shamtest::asserts().assert_float_equal(name, sycl::distance(sycl_ret ,check_val),0,1e-6);

}


template<class T,class Fct> f64 bench_reduction(Fct && red_fct, const u32 & size_test,std::vector<T> & vals){
    

    T sycl_ret;

    auto & q = shamsys::instance::get_compute_queue();

    shambase::Timer t;

    {
        sycl::buffer<T> buf {vals.data(),size_test};

        q.submit([&](sycl::handler & cgh){
            sycl::accessor a {buf,sycl::read_only};
            cgh.single_task([](){});
        });

        q.wait();

        t.start();

        

        sycl_ret = red_fct(q, buf, 0, size_test);

        q.wait();

        t.end();

    }


    return t.nanosec*1e-9;

}







template<class T>
void unit_test_reduc(){

    //unit_test_reduc<f64>("reduction : manual wg=32",
    //    [](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id) -> T {
    //        return shamalgs::reduction::details::GroupReduction<T, 32>::sum(q, buf1, start_id, end_id);
    //    }
    //);
    //unit_test_reduc<f64>("reduction : sycl2020", 
    //    [](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id) -> T {
    //        return shamalgs::reduction::details::SYCL2020<T>::sum(q, buf1, start_id, end_id);
    //    }
    //);
    unit_test_reduc<f64>("reduction : main",
        [](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id) -> T {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        }
    );

}

TestStart(Unittest, "shamalgs/reduction/sum", reduc_kernel_utest, 1){
    unit_test_reduc<f64>();
}










constexpr u32 lim_bench = 1e8;

template<class T,class Fct>
void bench_mark_indiv(std::string name,std::vector<T> & vals, Fct && fct){

    logger::info_ln("ShamrockTest","testing :",name);
    
    std::vector<f64> test_sz;
    for(f64 i = 16; i < lim_bench; i*=1.1){
        test_sz.push_back(i);
    }

    auto & res = shamtest::test_data().new_dataset(name);

    std::vector<f64> results;

    for(const f64 & sz : test_sz){
        logger::debug_ln("ShamrockTest","N=",sz);
        results.push_back(bench_reduction<T>(fct,u32(sz),vals));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);
    
}


template<class T> void bench_type(std::string Tname){

    std::vector<T> vals;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 1);

    for(u32 i = 0; i < lim_bench; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }


    #ifdef SYCL_COMP_DPCPP
    bench_mark_indiv<T>(Tname + " manual : wg=2",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,2>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=4",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,4>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=8",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,8>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=16",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,16>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=32",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,32>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=64",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,64>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=128",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,128>::sum(q, buf1, start_id, end_id);
    });

    bench_mark_indiv<T>(Tname + " sycl2020 reduction",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::SYCL2020<T>::sum(q, buf1, start_id, end_id);
    });
    #endif


    #ifdef SYCL_COMP_OPENSYCL
    bench_mark_indiv<T>(Tname + " manual : wg=2",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,2>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=4",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,4>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=8",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,8>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=16",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,16>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=32",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,32>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=64",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,64>::sum(q, buf1, start_id, end_id);
    });
    bench_mark_indiv<T>(Tname + " manual : wg=128",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
        return shamalgs::reduction::details::GroupReduction<T,128>::sum(q, buf1, start_id, end_id);
    });

    //bench_mark_indiv<T>(Tname + " sycl2020 reduction",vals,[](sycl::queue & q, sycl::buffer<T> & buf1, u32 start_id, u32 end_id){
    //    return shamalgs::reduction::details::SYCL2020<T>::sum(q, buf1, start_id, end_id);
    //});
    #endif
}

TestStart(Benchmark, "shamalgs/reduction/sum", reduc_kernel_bench, 1){

    bench_type<f64>("f32");
    bench_type<f64>("f64");

}

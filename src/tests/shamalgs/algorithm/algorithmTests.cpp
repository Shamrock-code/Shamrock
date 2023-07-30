// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sortTests.hpp"
#include "shamalgs/algorithm.hpp"

TestStart(Unittest, "shamalgs/algorithm/sort_by_key", test_sort_by_key_func, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );
    test.check();
}


TestStart(Benchmark, "shamalgs/algorithm/sort_by_key:benchmark", 
    test_sort_by_key_func_benchmark, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}

TestStart(Unittest, "shamalgs/algorithm/index_remap", test_index_remap_func, 1){

    TestIndexRemap<u32>(shamalgs::algorithm::index_remap<u32>).check();

}
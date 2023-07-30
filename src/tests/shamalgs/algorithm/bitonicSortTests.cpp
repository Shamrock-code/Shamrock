// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/algorithm/radixSortOnesweep.hpp"
#include "sortTests.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"

TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_legacy", test_bitonic_sort_legacy, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_legacy
        );
    test.check();
}




TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_updated", test_bitonic_sort_updated, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,16>
        );
    test.check();
}


TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_fallback", test_bitonic_sort_fallback, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback
        );
    test.check();
}

/*
TestStart(Unittest, "shamalgs/algorithm/details/sort_by_key_radix_onesweep_v3", test_sort_by_key_radix_onesweep_v3, 1){
    
    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,16,2>
        );
    test.check();
}
*/

TestStart(Benchmark, "shamalgs/algorithm/details/bitonicSorts:benchmark", 
    test_bitonic_sort_legacy_benchmark, 1){

    
    /*
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_legacy
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic legacy (u32, u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,8>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,8)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,16>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,16)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    /*
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,32>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,1>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,1");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,2>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,2");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,4>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,4");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,8>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,8");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    /* disabled
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,8>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,8)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,16>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,16)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,32>
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic fallback (u32,u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    if(false){
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );
    
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic public (u32,u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }


}
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/random.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

template<class T>
inline void test(){
    u32 wsize = shamcomm::world_size();
    u32 wrank = shamcomm::world_rank();
    std::vector<T> vectest = shamalgs::random::mock_vector<T>(0x1111,wsize);

    T sum = shamalgs::collective::allreduce_sum(vectest[wrank]);

    T ref_sum{};

    for(T tmp : vectest){
        ref_sum += tmp;
    }

    shamtest::asserts().assert_bool("same value", shambase::vec_equals(sum,ref_sum));
}


TestStart(Unittest, "shamalgs/collective/reduction/allreduce_sum", testsallreducesum, -1){
    test<f32>();
    test<u32>();
    test<f64_2>();
    test<f64_3>();
}
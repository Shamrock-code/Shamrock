#include "test_tree.hpp"
#include "../unit_test_handler.hpp"
#include "../../tree/kernels/key_morton_sort.hpp"
#include "../../sys/sycl_handler.hpp"
#include "../../utils/string_utils.hpp"
#include <algorithm>
#include <random>

#if defined(PRECISION_MORTON_DOUBLE)
    #define FILL(i) morton_list[i] = 18446744073709551615ul
#else
    #define FILL(i) morton_list[i] = 4294967295u
#endif

#define size_test 128*128

void run_tests_morton_code_sort(){


    if(unit_test::test_start("tree/key_morton_sort.hpp", false)){

        std::vector<u_morton> morton_list;

        for(u32 i = 0; i < size_test; i++){
            morton_list.push_back(i);
        }

        FILL(7);
        FILL(15);
        FILL(53);
        FILL(371);
        FILL(54);
        FILL(566);
        FILL(647);
        FILL(1000);
        FILL(888);
        FILL(666);

        shuffle (morton_list.begin(), morton_list.end(), std::default_random_engine(647915));

        std::vector<u_morton> unsorted(morton_list.size());

        std::copy(morton_list.begin(), morton_list.end(),unsorted.begin());

        {
            sycl::buffer<u_morton> buf_morton(morton_list);
            sycl::buffer<u32> buf_index(morton_list.size());

            sycl_sort_morton_key_pair(
                queue,
                size_test,
                & buf_index,
                & buf_morton
                );

        }

        std::sort(unsorted.begin(), unsorted.end());


        for(u32 i = 0; i < size_test; i++){
            unit_test::test_assert(("index [" +format("%d",i)+ "]").c_str(),  unsorted[i]  , morton_list[i]);
        }



    }unit_test::test_end();

}
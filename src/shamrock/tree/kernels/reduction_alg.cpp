// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file reduction_alg.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */


#include "reduction_alg.hpp"
#include <algorithm>
#include <memory>
#include <vector>

#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shambase/exception.hpp"
#include "shambackends/sycl.hpp"
#include "shambase/string.hpp"
#include "shambase/integer_sycl.hpp"

class Kernel_generate_split_table_morton32;
class Kernel_generate_split_table_morton64;

template<class u_morton, class kername, class split_int>
void sycl_generate_split_table(
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    std::unique_ptr<sycl::buffer<split_int>> &buf_split_table
) {

    sycl::range<1> range_morton_count{morton_count};

    queue.submit([&](sycl::handler &cgh) {

        sycl::accessor m {*buf_morton, cgh, sycl::read_only};
        sycl::accessor split_out {*buf_split_table,cgh,sycl::write_only,sycl::no_init};

        cgh.parallel_for<kername>(range_morton_count, [=](sycl::item<1> item) {
            u32 i = (u32)item.get_id(0);

            if (i > 0) {
                if (m[i - 1] != m[i]) {
                    split_out[i] = 1;
                } else {
                    split_out[i] = 0;
                }
            } else {
                split_out[i] = 1;
            }
        });
    });
}



/*
Godbolt snippet for testing

#include <cstdio>
#include <array>

template<int Na, int Nb>
void print(std::array<bool, Na> & foo, std::array<int, Nb> cursors){
    for (int i = 0; i < Na; ++i)
        printf("%d", foo[i]);
    printf("\n");

    for(int i : cursors){
        for(int j = 0 ; j < i ; j++) printf(" ");
        printf("|\n");
    }
}


int main(){
    auto a  = std::array<bool,11>{1,0,1,1,1,0,0,1,0,1,0};

    int i = 7;
    int before1 = i - 1;
    while (!a[before1])
        before1--;

    int before2 = before1 - 1;
    while (!a[before2])
        before2--;


    int next1 = i +1;
    while (!a[next1])
        next1++;

    print<11,4>(a, {next1,i,before1, before2});
}


*/

class Kernel_iterate_reduction_morton32;
class Kernel_iterate_reduction_morton64;

//#define OLD_BEHAVIOR
#define NEW_BEHAVIOR

#ifdef NEW_BEHAVIOR
#define OFFSET 
#endif

#ifdef OLD_BEHAVIOR
#define OFFSET + 1
#endif



template<class u_morton, class kername, class split_int>
void sycl_reduction_iteration(
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    std::unique_ptr<sycl::buffer<split_int>> &buf_split_table_in,
    std::unique_ptr<sycl::buffer<split_int>> &buf_split_table_out
) {

    
    sycl::range<1> range_morton_count{morton_count};

    queue.submit([&](sycl::handler &cgh) {
        u32 _morton_cnt = morton_count;

        sycl::accessor m{*buf_morton, cgh, sycl::read_only};
        sycl::accessor split_in{*buf_split_table_in,cgh , sycl::read_only};
        sycl::accessor split_out{*buf_split_table_out,cgh , sycl::write_only, sycl::no_init};

        cgh.parallel_for<kername>(range_morton_count, [=](sycl::item<1> item) {
            int i = item.get_id(0);

            auto DELTA = [=](i32 x, i32 y) { return shambase::karras_delta(x, y, _morton_cnt, m); };

            // find index of preceding i-1 non duplicate morton code
            u32 before1 = i - 1;
            while (before1 <= _morton_cnt - 1 && !split_in[before1 OFFSET])
                before1--;

            // find index of preceding i-2 non duplicate morton code
            // safe bc delta(before1,before2) return -1 if any of the 2 are -1 because of order
            u32 before2 = before1 - 1;
            while (before2 <= _morton_cnt - 1 && !split_in[before2 OFFSET])
                before2--;

            // find index of next i+1 non duplicate morton code
            u32 next1 = i + 1;
            while (next1 <= _morton_cnt - 1 && !split_in[next1])
                next1++;

            int delt_0  = DELTA(i, next1);
            int delt_m  = DELTA(i, before1);
            int delt_mm = DELTA(before1, before2);

            if (!(delt_0 < delt_m && delt_mm < delt_m) && split_in[i]) {
                split_out[i] = 1;
            } else {
                split_out[i] = 0;
            }
        });
    });
}


void update_morton_buf(
    sycl::queue &queue,
    u32 len,u32 val_ins,
    sycl::buffer<u32>& buf_src, 
    std::unique_ptr<sycl::buffer<u32>> & buf_reduc_index_map
){

    sycl::range<1> range_morton_count{len+2};

    queue.submit([&](sycl::handler &cgh) {
        u32 _len = len;
        u32 val = val_ins;

        sycl::accessor src{buf_src, cgh, sycl::read_only};
        sycl::accessor dest{*buf_reduc_index_map,cgh , sycl::write_only, sycl::no_init};

        cgh.parallel_for(range_morton_count, [=](sycl::item<1> item) {

            if(item.get_linear_id() < _len){
                dest[item] = src[item];
            }else if (item.get_linear_id() == _len) {
                dest[item] = val;
            }else if (item.get_linear_id() == _len+1) {
                dest[item] = 0;
            }

        });

    });

}


template<class split_int>
void make_indexmap(
    sycl::queue &queue,
    u32 morton_count,
    u32 &morton_leaf_count,
    std::unique_ptr<sycl::buffer<split_int>> & buf_split_table,
    std::unique_ptr<sycl::buffer<u32>> & buf_reduc_index_map
){


    auto [buf,len] = shamalgs::numeric::stream_compact(queue, *buf_split_table, morton_count);

    morton_leaf_count = len;

    buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(morton_leaf_count+2);


    if(buf){
        update_morton_buf(queue, len,morton_count, *buf, buf_reduc_index_map);
    }else{
        throw shambase::make_except_with_loc<std::runtime_error>("this result shouldn't be null");
    }

    if constexpr (false){
        std::vector<u32> reduc_index_map;

        u32 leafs = 0;

        {
            sycl::host_accessor acc {*buf_split_table, sycl::read_only};

            // reduc_index_map.reserve(split_count);
            for (unsigned int i = 0; i < morton_count; i++) {
                if (acc[i]) {
                    reduc_index_map.push_back(i);
                    leafs++;
                }
            }
            reduc_index_map.push_back(morton_count);
            // for one cell mode the last range is inverted to avoid iteration
            reduc_index_map.push_back(0); 
        }

        {
             if(leafs != morton_leaf_count){
                throw shambase::make_except_with_loc<std::runtime_error>("difference");
            }
        
            sycl::host_accessor dest{*buf_reduc_index_map , sycl::read_only};
        
            for (unsigned int i = 0; i < morton_leaf_count+2; i++) {
                if(dest[i] != reduc_index_map[i]){
                    throw shambase::make_except_with_loc<std::runtime_error>(shambase::format("difference i = {}, {} != {}",i, dest[i] , reduc_index_map[i]));
                }
            }
        }

        buf_reduc_index_map =
            std::make_unique<sycl::buffer<u32>>(shamalgs::memory::vector_to_buf(reduc_index_map));

    }
}


template<class u_morton, class kername_split, class kername_reduc_it>
void reduction_alg_impl(
    // in
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    u32 reduction_level,
    // out
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    u32 &morton_leaf_count
) {

    auto buf_split_table1 = std::make_unique<sycl::buffer<u32>>(morton_count);
    auto buf_split_table2 = std::make_unique<sycl::buffer<u32>>(morton_count);

    sycl_generate_split_table<u_morton, kername_split>(
        queue, morton_count, buf_morton, buf_split_table1
    );

    for (unsigned int iter = 1; iter <= reduction_level; iter++) {

        if (iter % 2 == 0) {
            sycl_reduction_iteration<u_morton, kername_reduc_it>(
                queue, morton_count, buf_morton, buf_split_table2, buf_split_table1
            );
        } else {
            sycl_reduction_iteration<u_morton, kername_reduc_it>(
                queue, morton_count, buf_morton, buf_split_table1, buf_split_table2
            );
        }

    }

    std::unique_ptr<sycl::buffer<u32>> buf_split_table;
    if ((reduction_level) % 2 == 0) {
        buf_split_table = std::move(buf_split_table1);
    } else {
        buf_split_table = std::move(buf_split_table2);
    }

    make_indexmap(queue,morton_count, morton_leaf_count, buf_split_table,buf_reduc_index_map);

}

template<>
void reduction_alg<u32>(
    // in
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_morton,
    u32 reduction_level,
    // out
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    u32 &morton_leaf_count
) {
    reduction_alg_impl<
        u32,
        Kernel_generate_split_table_morton32,
        Kernel_iterate_reduction_morton32>(
        queue, morton_count, buf_morton, reduction_level, buf_reduc_index_map, morton_leaf_count
    );
}

template<>
void reduction_alg<u64>(
    // in
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u64>> &buf_morton,
    u32 reduction_level,
    // out
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    u32 &morton_leaf_count
) {
    reduction_alg_impl<
        u64,
        Kernel_generate_split_table_morton64,
        Kernel_iterate_reduction_morton64>(
        queue, morton_count, buf_morton, reduction_level, buf_reduc_index_map, morton_leaf_count
    );
}

class Kernel_remap_morton_code_morton32;
class Kernel_remap_morton_code_morton64;

template<class u_morton, class kername>
void __sycl_morton_remap_reduction(
    // in
    sycl::queue &queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    // out
    std::unique_ptr<sycl::buffer<u_morton>> &buf_leaf_morton
) {
    sycl::range<1> range_remap_morton{morton_leaf_count};

    queue.submit([&](sycl::handler &cgh) {
        auto id_remaped = buf_reduc_index_map->get_access<sycl::access::mode::read>(cgh);
        auto m          = buf_morton->template get_access<sycl::access::mode::read>(cgh);
        auto m_remaped =
            buf_leaf_morton->template get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<kername>(range_remap_morton, [=](sycl::item<1> item) {
            int i = item.get_id(0);

            m_remaped[i] = m[id_remaped[i]];
        });
    });
}

template<>
void sycl_morton_remap_reduction<u32>(
    // in
    sycl::queue &queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u32>> &buf_morton,
    // out
    std::unique_ptr<sycl::buffer<u32>> &buf_leaf_morton
) {
    __sycl_morton_remap_reduction<u32, Kernel_remap_morton_code_morton32>(
        queue, morton_leaf_count, buf_reduc_index_map, buf_morton, buf_leaf_morton
    );
}

template<>
void sycl_morton_remap_reduction<u64>(
    // in
    sycl::queue &queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u64>> &buf_morton,
    // out
    std::unique_ptr<sycl::buffer<u64>> &buf_leaf_morton
) {
    __sycl_morton_remap_reduction<u64, Kernel_remap_morton_code_morton64>(
        queue, morton_leaf_count, buf_reduc_index_map, buf_morton, buf_leaf_morton
    );
}

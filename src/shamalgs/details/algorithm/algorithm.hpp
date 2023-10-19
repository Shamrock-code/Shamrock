// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file algorithm.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief main include file for the shamalgs algorithms
 * 
 */


#include "shambase/type_aliases.hpp"
#include "shambase/sycl_vec_aliases.hpp"

#include <shambackends/backends/sycl/sycl.hpp>

/**
 * @brief namespace to store algorithms implemented by shamalgs
 * 
 */
namespace shamalgs::algorithm {



    //template<class T>
    //void sort(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    /**
     * @brief Sort the buffer according to the key order
     * 
     * @tparam T 
     * @param q 
     * @param buf_key 
     * @param buf_values 
     * @param len 
     */
    template<class Tkey, class Tval>
    void sort_by_key(sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);


    
    /**
     * @brief generate a buffer from a lambda expression based on the indexes
     * 
     * @tparam Fct 
     * @param q 
     * @param len 
     * @param func 
     * @return sycl::buffer<typename std::invoke_result_t<Fct,u32>> 
     */
    template<class Fct>
    inline sycl::buffer<typename std::invoke_result_t<Fct,u32>> gen_buffer_device(sycl::queue & q, u32 len, Fct && func){

        using ret_t = typename std::invoke_result_t<Fct,u32>;

        sycl::buffer<ret_t> ret(len);

        q.submit([&](sycl::handler &cgh) {

            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>(len),
                [=](sycl::item<1> item) {
                    out[item] = func(item.get_linear_id());
                }
            );
        });

        return std::move(ret);
    }


    /**
     * @brief remap a buffer according to a given index map
     * result[i] = result[index_map[i]]
     * 
     * This function can be used to apply a sort to another object
     * 
     * @tparam T type of the buffer
     * @param q the sycl queue
     * @param buf the buffer to apply the remapping on
     * @param index_map the index map 
     * @param len length of the index map
     */
    template<class T>
    sycl::buffer<T> index_remap(
        sycl::queue &q, sycl::buffer<T> &source_buf, sycl::buffer<u32> &index_map, u32 len);
    
    /**
     * @brief remap a buffer (with multiple variable per index) according to a given index map
     * result[i] = result[index_map[i]]
     * 
     * This function can be used to apply a sort to another object
     * 
     * @tparam T type of the buffer
     * @param q the sycl queue
     * @param buf the buffer to apply the remapping on
     * @param index_map the index map 
     * @param len length of the index map
     * @param nvar the number of variable per index
     */
    template<class T>
    sycl::buffer<T> index_remap_nvar(
        sycl::queue &q, sycl::buffer<T> &source_buf, sycl::buffer<u32> &index_map, u32 len, u32 nvar);

    

    /**
     * @brief generate a buffer such that for i in [0,len[, buf[i] = i 
     * 
     * @param q the queue to run on
     * @param len length of the buffer to generate
     * @return sycl::buffer<u32> the returned buffer
     */
    sycl::buffer<u32> gen_buffer_index(sycl::queue & q , u32 len);


    

}
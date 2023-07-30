// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "numericFallback.hpp"
#include "shamalgs/memory/memory.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    sycl::buffer<T> exclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){

        sycl::buffer<T> ret_buf (len);

        T accum {0};

        {
            sycl::host_accessor acc_src {buf1, sycl::read_only};
            sycl::host_accessor acc_res {ret_buf, sycl::write_only, sycl::no_init};

            for(u32 idx = 0; idx < len; idx ++){
                
                acc_res[idx] = accum;
                accum += acc_src[idx];

            }
        }

        return std::move(ret_buf);

    }

    template<class T>
    sycl::buffer<T> inclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){

        sycl::buffer<T> ret_buf (len);

        T accum {0};

        {
            sycl::host_accessor acc_src {buf1, sycl::read_only};
            sycl::host_accessor acc_res {ret_buf, sycl::write_only, sycl::no_init};

            for(u32 idx = 0; idx < len; idx ++){
                
                accum += acc_src[idx];
                acc_res[idx] = accum;

            }
        }

        return std::move(ret_buf);

    }

    template<class T>
    void exclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){

        T accum {0};

        {
            sycl::host_accessor acc_src {buf1, sycl::read_write};

            for(u32 idx = 0; idx < len; idx ++){

                T val = accum;
                
                accum += acc_src[idx];

                acc_src[idx] = val;

            }
        }

    }

    template<class T>
    void inclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){


        T accum {0};

        {
            sycl::host_accessor acc_src {buf1, sycl::read_write};

            for(u32 idx = 0; idx < len; idx ++){
                
                accum += acc_src[idx];
                acc_src[idx] = accum;

            }
        }

    }

    template
    sycl::buffer<u32> inclusive_sum_fallback(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template
    sycl::buffer<u32> exclusive_sum_fallback(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template
    void exclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template
    void inclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);


    std::tuple<std::optional<sycl::buffer<u32>>, u32> stream_compact_fallback(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len){

        std::vector<u32> idxs ;


        {
            sycl::host_accessor acc_src {buf_flags, sycl::read_only};

            for(u32 idx = 0; idx < len; idx ++){
                
                if(acc_src[idx]){
                    idxs.push_back(idx);
                }

            }
        }

        if(idxs.empty()){
            return {{},0};
        }

        return {memory::vec_to_buf(idxs), idxs.size()};

    }


}
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "fallbackReduction.hpp"
#include "shambase/sycl_utils.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    T _int_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        T accum;

        {
            sycl::host_accessor acc {buf1, sycl::read_only};

            for(u32 idx = start_id; idx < end_id; idx ++){
                if(idx == start_id){
                    accum = acc[idx];
                }else{
                    accum += acc[idx];
                }
            }
        }

        return accum;
    }

    template<class T>
    T _int_min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        T accum;

        {
            sycl::host_accessor acc {buf1, sycl::read_only};

            for(u32 idx = start_id; idx < end_id; idx ++){
                if(idx == start_id){
                    accum = acc[idx];
                }else{
                    accum = shambase::sycl_utils::g_sycl_min(acc[idx], accum);
                }
            }
        }

        return accum;
    }

    template<class T>
    T _int_max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        T accum;

        {
            sycl::host_accessor acc {buf1, sycl::read_only};

            for(u32 idx = start_id; idx < end_id; idx ++){
                if(idx == start_id){
                    accum = acc[idx];
                }else{
                    accum = shambase::sycl_utils::g_sycl_max(acc[idx], accum);
                }
            }
        }

        return accum;
    }





    
    template<class T>
    T FallbackReduction<T>::sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_sum(q, buf1, start_id, end_id);
        
    }

    template<class T>
    T FallbackReduction<T>::min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_min(q, buf1, start_id, end_id);
        
    }

    template<class T>
    T FallbackReduction<T>::max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_max(q, buf1, start_id, end_id);
        
    }




    #define XMAC_TYPES \
    X(f32   ) \
    X(f32_2 ) \
    X(f32_3 ) \
    X(f32_4 ) \
    X(f32_8 ) \
    X(f32_16) \
    X(f64   ) \
    X(f64_2 ) \
    X(f64_3 ) \
    X(f64_4 ) \
    X(f64_8 ) \
    X(f64_16) \
    X(u32   ) \
    X(u64   ) \
    X(u32_3 ) \
    X(u64_3 ) \
    X(i64_3 ) \
    X(i64 )

    #define X(_arg_)\
    template struct FallbackReduction<_arg_>;

    XMAC_TYPES
    #undef X

}
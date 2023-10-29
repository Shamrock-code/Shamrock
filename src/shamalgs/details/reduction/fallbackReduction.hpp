// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file fallbackReduction.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "aliases.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    struct FallbackReduction{

        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    };

    template<class T>
    inline T _int_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
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
    inline T _int_min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        T accum;

        {
            sycl::host_accessor acc {buf1, sycl::read_only};

            for(u32 idx = start_id; idx < end_id; idx ++){
                if(idx == start_id){
                    accum = acc[idx];
                }else{
                    accum = sham::min(acc[idx], accum);
                }
            }
        }

        return accum;
    }

    template<class T>
    inline T _int_max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        T accum;

        {
            sycl::host_accessor acc {buf1, sycl::read_only};

            for(u32 idx = start_id; idx < end_id; idx ++){
                if(idx == start_id){
                    accum = acc[idx];
                }else{
                    accum = sham::max(acc[idx], accum);
                }
            }
        }

        return accum;
    }





    
    template<class T>
    inline T FallbackReduction<T>::sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_sum(q, buf1, start_id, end_id);
        
    }

    template<class T>
    inline T FallbackReduction<T>::min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_min(q, buf1, start_id, end_id);
        
    }

    template<class T>
    inline T FallbackReduction<T>::max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        
        return _int_max(q, buf1, start_id, end_id);
        
    }

} // namespace shamalgs::reduction::details
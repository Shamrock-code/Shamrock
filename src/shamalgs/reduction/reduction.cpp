// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "reduction.hpp"

#include "shamalgs/reduction/details/sycl2020reduction.hpp"
#include "shamalgs/reduction/details/groupReduction.hpp"
#include "shamalgs/reduction/details/fallbackReduction.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include <hipSYCL/sycl/buffer.hpp>
#include <hipSYCL/sycl/handler.hpp>
#include <hipSYCL/sycl/libkernel/accessor.hpp>

namespace shamalgs::reduction {


    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::SYCL2020<T>::sum(q, buf1, start_id, end_id);
    }

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::FallbackReduction<T>::max(q, buf1, start_id, end_id);
    }

    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        return details::FallbackReduction<T>::min(q, buf1, start_id, end_id);
    }

    template<class T>
    shambase::VecComponent<T> dot_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        sycl::buffer<shambase::VecComponent<T>> ret_data_base(end_id - start_id);

        q.submit([&](sycl::handler & cgh){
            sycl::accessor acc_dot {ret_data_base, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc{buf1, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>{end_id - start_id}, [=](sycl::item<1> it){
                const T tmp = acc[it];
                acc_dot[it] = shambase::sycl_utils::g_sycl_dot(tmp,tmp);
            });
        });

        return sum(q, ret_data_base, 0, end_id - start_id);
    }

    bool is_all_true(sycl::buffer<u8> & buf,u32 cnt){

        //TODO do it on GPU pleeeaze

        bool res = true;
        {
            sycl::host_accessor acc{buf, sycl::read_only};

            for (u32 i = 0; i < cnt; i++) {
                res = res && (acc[i] != 0);
            }
        }

        return res;

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
    X(u64_3 )

    #define X(_arg_) \
    template _arg_ sum(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);\
    template shambase::VecComponent<_arg_> dot_sum(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);\
    template _arg_ max(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);\
    template _arg_ min(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);

    XMAC_TYPES
    #undef X

} // namespace shamalgs::reduction
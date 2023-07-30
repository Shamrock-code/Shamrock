// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sycl2020reduction.hpp"
#include "shamalgs/memory.hpp"


namespace shamalgs::reduction::details {
    template<class T, class Op>
    inline T
    reduce_sycl_2020(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id, Op op) {

        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);
        shamalgs::memory::write_with_offset_into(buf_int, buf1, start_id, len);

        sycl::buffer<T> recov{1};

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor global_mem{buf_int, cgh, sycl::read_only};

#ifdef SYCL_COMP_DPCPP
            auto reduc = sycl::reduction(recov, cgh, op);
#else
            sycl::accessor acc_rec{recov, cgh, sycl::write_only, sycl::no_init};
            auto reduc = sycl::reduction(acc_rec, op);
#endif

            cgh.parallel_for(sycl::range<1>{len}, reduc, [=](sycl::id<1> idx, auto &sum) {
                sum.combine(global_mem[idx]);
            });
        });

        T rec;
        {
            sycl::host_accessor acc{recov, sycl::read_only};
            rec = acc[0];
        }

        return rec;
    }

    template<class T>
    T SYCL2020<T>::sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        #ifdef SYCL_COMP_DPCPP
        return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::plus<>{});
        #endif

        #ifdef SYCL_COMP_OPENSYCL
        return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::plus<T>{});
        #endif
    }

    //template<class T>
    //T SYCL2020<T>::min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
    //    #ifdef SYCL_COMP_DPCPP
    //    return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::minimum<>{});
    //    #endif
//
    //    #ifdef SYCL_COMP_OPENSYCL
    //    return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::minimum<T>{});
    //    #endif
    //}
//
    //template<class T>
    //T SYCL2020<T>::max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
    //    #ifdef SYCL_COMP_DPCPP
    //    return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::maximum<>{});
    //    #endif
//
    //    #ifdef SYCL_COMP_OPENSYCL
    //    return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::maximum<T>{});
    //    #endif
    //}



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

    #define X(_arg_)\
    template struct SYCL2020<_arg_>;

    XMAC_TYPES
    #undef X

} // namespace shamalgs::reduction::details
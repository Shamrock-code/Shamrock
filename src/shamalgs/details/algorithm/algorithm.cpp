// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file algorithm.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "algorithm.hpp"
#include "bitonicSort.hpp"

namespace shamalgs::algorithm {

    template<class Tkey, class Tval>
    void sort_by_key(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    ) {

        if(len < 5e3){
            details::sort_by_key_bitonic_fallback(q, buf_key, buf_values, len);
        }else{
            details::sort_by_key_bitonic_updated<Tkey, Tval, 16>(q, buf_key, buf_values, len);
        }

    }

    template void
    sort_by_key(sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void
    sort_by_key(sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    sycl::buffer<u32> gen_buffer_index(sycl::queue &q, u32 len) {
        return gen_buffer_device(q, len, [](u32 i) -> u32 { return i; });
    }








    template<class T>
    sycl::buffer<T> index_remap(
        sycl::queue &q, sycl::buffer<T> &buf, sycl::buffer<u32> &index_map, u32 len) {

        sycl::buffer<T> ret(len);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor in{buf, cgh, sycl::read_only};
            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor permut{index_map, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                out[item] = in[permut[item]];
            });
        });

        return std::move(ret);
    }

    template<class T>
    sycl::buffer<T> index_remap_nvar(
        sycl::queue &q, sycl::buffer<T> &buf, sycl::buffer<u32> &index_map, u32 len, u32 nvar) {

        sycl::buffer<T> ret(len*nvar);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor in{buf, cgh, sycl::read_only};
            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor permut{index_map, cgh, sycl::read_only};

            u32 nvar_loc = nvar;

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                u32 in_id  = permut[item] * nvar_loc;
                u32 out_id = item.get_linear_id() * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    out[out_id + a] = in[in_id + a];
                }
            });
        });

        return std::move(ret);
    }






#define XMAC_TYPES                                                                                 \
    X(f32)                                                                                         \
    X(f32_2)                                                                                       \
    X(f32_3)                                                                                       \
    X(f32_4)                                                                                       \
    X(f32_8)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64)                                                                                         \
    X(f64_2)                                                                                       \
    X(f64_3)                                                                                       \
    X(f64_4)                                                                                       \
    X(f64_8)                                                                                       \
    X(f64_16)                                                                                      \
    X(u32)                                                                                         \
    X(u64)                                                                                         \
    X(u32_3)                                                                                       \
    X(u64_3)                                                                                       \
    X(i64_3)

#define X(_arg_)                                                                                   \
    template sycl::buffer<_arg_> index_remap(\
        sycl::queue &q, sycl::buffer<_arg_> &buf, sycl::buffer<u32> &index_map, u32 len);                                                                                             \
                                                                                                   \
    template sycl::buffer<_arg_> index_remap_nvar(\
        sycl::queue &q, sycl::buffer<_arg_> &buf, sycl::buffer<u32> &index_map, u32 len, u32 nvar);

    XMAC_TYPES

    #undef X

} // namespace shamalgs::algorithm
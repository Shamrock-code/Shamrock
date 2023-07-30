// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/details/numeric/streamCompactExclScan.hpp"

namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){
        return details::exclusive_sum_atomic_decoupled_v5<T, 512>(q, buf1, len);
    }

    template<class T>
    sycl::buffer<T> inclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){
        return details::inclusive_sum_fallback(q, buf1, len);
    }


    





    template<class T>
    void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){
        buf1 = details::exclusive_sum_atomic_decoupled_v5<T, 256>(q, buf1, len);
    }

    template<class T>
    void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf1, u32 len){
        buf1 = details::inclusive_sum_fallback(q, buf1, len);
    }


    

    template sycl::buffer<u32> exclusive_sum(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);
    template sycl::buffer<u32> inclusive_sum(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);
    template void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);





    std::tuple<std::optional<sycl::buffer<u32>>, u32> stream_compact(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len){
        return details::stream_compact_excl_scan(q, buf_flags, len);
    };

} // namespace shamalgs::numeric

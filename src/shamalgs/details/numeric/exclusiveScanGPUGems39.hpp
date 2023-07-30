// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_1(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);
    
    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_2(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);
    
    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_3(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);
    

    
} // namespace shamalgs::numeric::details

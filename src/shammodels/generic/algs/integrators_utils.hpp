// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//%Impl status : Good

#pragma once

/**
 * @file integrators_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include <iostream>

template<class T, class flt>
inline void field_advance_time(sycl::queue & queue, sycl::buffer<T> & buf_val, sycl::buffer<T> & buf_der, sycl::range<1> elem_range, flt dt){

    auto ker_advance_time = [&](sycl::handler &cgh) {
        auto acc_u = buf_val.template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_du = buf_der.template get_access<sycl::access::mode::read>(cgh);

        // Executing kernel
        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            T du = acc_du[item];

            acc_u[item] = acc_u[item] + (dt) * (du);

        });
    };

    queue.submit(ker_advance_time);

}
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

/**
 * @file interface_selector.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shamsys/legacy/sycl_handler.hpp"
#include <tuple>

template <class vectype, class field_type> class InterfaceSelector_SPH {
  public:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

    static std::tuple<vectype, vectype> get_compute_box_sz(vectype compute_box_min, vectype compute_box_max,
                                                           field_type compute_box_field_val,
                                                           field_type neighbourg_box_field_val) {
        field_type h = sycl::max(compute_box_field_val, neighbourg_box_field_val);
        return {compute_box_min - h, compute_box_max + h};
    }

    static std::tuple<vectype, vectype> get_neighbourg_box_sz(vectype neighbourg_box_min, vectype neighbourg_box_max,
                                                              field_type compute_box_field_val,
                                                              field_type neighbourg_box_field_val) {
        return {neighbourg_box_min, neighbourg_box_max};
    }

#pragma clang diagnostic pop
};

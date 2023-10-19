// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file particleGen.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/sycl_utils.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include <functional>
#include <vector>

namespace shamrock::sph {



    template<class flt>
    inline sycl::vec<flt, 3> get_box_dim(flt r_particle, u32 xcnt, u32 ycnt, u32 zcnt){

        using vec3 = sycl::vec<flt, 3>;

        u32 im = xcnt;
        u32 jm = ycnt;
        u32 km = zcnt;

        auto get_pos = [&](u32 i, u32 j, u32 k) -> vec3{
            vec3 r_a = {
                2*i + ((j+k) % 2),
                sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                2*sycl::sqrt(6.)*k/3
            };

            r_a *= r_particle;

            return r_a;
        };

        return get_pos(im,jm,km);
    }

    template<class flt> 
    inline std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> get_ideal_fcc_box(flt r_particle, 
        std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box){

        using vec3 = sycl::vec<flt, 3>;

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);

        vec3 box_dim = box_max - box_min;

        vec3 iboc_dim = (box_dim / 
            vec3({
                2,
                sycl::sqrt(3.),
                2*sycl::sqrt(6.)/3
            }))/r_particle;

        u32 i = iboc_dim.x();
        u32 j = iboc_dim.y();
        u32 k = iboc_dim.z();

        //std::cout << "get_ideal_box_idim :" << i << " " << j << " " << k << std::endl;

        i -= i%2;
        j -= j%2;
        k -= k%2;

        vec3 m1 = get_box_dim(r_particle, i, j, k);

        return {box_min, box_min + m1};

    }

    template<class Tscal>
    inline sycl::vec<Tscal, 3> get_fcc_pos(u32 i, u32 j, u32 k){
        return {
                2*i + ((j+k) % 2),
                sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                2*sycl::sqrt(6.)*k/3
            };
    }    

}
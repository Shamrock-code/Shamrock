// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good


#pragma once

#include "aliases.hpp"
#include "shamalgs/random/random.hpp"
#include "shambase/type_aliases.hpp"
#include "shambase/sycl.hpp"

namespace generic::setup::generators {

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

        std::cout << "get_ideal_box_idim :" << i << " " << j << " " << k << std::endl;

        i -= i%2;
        j -= j%2;
        k -= k%2;

        vec3 m1 = get_box_dim(r_particle, i, j, k);

        return {box_min, box_min + m1};

    }


    


    template<class flt,class Tpred_select,class Tpred_pusher>
    inline void add_particles_fcc(
        flt r_particle, 
        std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box,
        Tpred_select && selector,
        Tpred_pusher && part_pusher ){
        
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

        std::cout << "part box size : (" << iboc_dim.x() << ", " << iboc_dim.y() << ", " << iboc_dim.z() << ")" << std::endl;
        u32 ix = std::ceil(iboc_dim.x());
        u32 iy = std::ceil(iboc_dim.y());
        u32 iz = std::ceil(iboc_dim.z());
        std::cout << "part box size : (" << ix << ", " << iy << ", " << iz << ")" << std::endl;

        if((iy % 2) != 0 && (iz % 2) != 0){
            std::cout << "Warning : particle count is odd on axis y or z -> this may lead to periodicity issues";
        }

        for(u32 i = 0 ; i < ix; i++){
            for(u32 j = 0 ; j < iy; j++){
                for(u32 k = 0 ; k < iz; k++){

                    vec3 r_a = {
                        2*i + ((j+k) % 2),
                        sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                        2*sycl::sqrt(6.)*k/3
                    };

                    r_a *= r_particle;
                    r_a += box_min;

                    if(selector(r_a)) part_pusher(r_a, r_particle);

                }
            }
        }


    }


    /**
     * @brief 
     * 
     * @tparam flt 
     * @tparam Tpred_pusher 
     * @param Npart number of particles
     * @param p randial power law surface density (default = 1)  sigma prop r^-p
     * @param rho_0 rho_0 volumic density (at r = 1)
     * @param m mass part
     * @param r_in inner cuttof
     * @param r_out outer cuttof
     * @param q T prop r^-q
     */
    template<class flt,class Tpred_pusher>
    inline void add_disc(
        u32 Npart,
        flt p,
        flt rho_0,
        flt m,
        flt r_in,
        flt r_out,
        flt q,
        Tpred_pusher && part_pusher 
    ){
        flt _2pi = 2*M_PI;


        flt K = _2pi*rho_0/m;
        flt c = 2-p;

        flt y = K*(r_out-r_in)/c;
        

        std::mt19937 eng(0x111);

        for(u32 i = 0 ;i < Npart; i++){

            flt r_1 = shamalgs::random::mock_value<flt>(eng,0, y);
            flt r_2 = shamalgs::random::mock_value<flt>(eng,0, _2pi);
            flt r_3 = shamalgs::random::mock_value<flt>(eng,0, 1);
            flt r_4 = shamalgs::random::mock_value<flt>(eng,0, 1);

            flt r = sycl::pow(
                sycl::pow(r_in, c) + c*r_1/K ,
                1/c);
            
            flt theta = r_2;

            flt u = sycl::sqrt(-2*sycl::log(r_3))*sycl::cos(_2pi*r_4);

            flt H = 0.1*sycl::pow(r,(flt)(3./2. - q/2));

            part_pusher(
                sycl::vec<flt, 3>({r*sycl::cos(theta),u*H,r*sycl::sin(theta)}), 
                0.1);

        }



    }


} // namespace generic::setup::generators
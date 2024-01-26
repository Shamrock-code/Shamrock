// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Model.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamunits/Names.hpp"
#include <vector>

using Tscal              = shambase::VecComponent<Tvec>;

inline void disc_analysis(Tvec xyz, Tvec vxyz,int npart, Tscal pmass, Tscal time,
                        int nbin, Tscal rmin, Tscal rmax, Tscal G, Tscal M_star) {
            
    Tvec lx
    Tscal dbin = (rmax-rmin)/real(nbin-1)

    do i=1,nbin
        bin(i)=rmin + real(i-1)*dbin
    enddo


        ninbin(:)=0
        lx(:)=0.0
        ly(:)=0.0
        lz(:)=0.0
        h_smooth(:)=0.0
        sigma(:)=0.0
        ecc=0.0
        angx = 0.0
        angy = 0.0
        angz = 0.0
        twist = 0.0
        twistprev = 0.0
        mu = G*M_star


     }
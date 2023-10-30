// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SinkPartStruct.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/sycl_utils/vectorProperties.hpp"
namespace shammodels::sph {

    template<class Tvec>
    struct SinkParticle {

        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        Tvec pos;
        Tvec velocity;
        Tvec sph_acceleration;
        Tvec ext_acceleration;
        Tscal mass;
        Tvec angular_momentum;
        Tscal accretion_radius;
    };

} // namespace shammodels::sph
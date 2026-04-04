// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RenderConfig.hpp

 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */


namespace shammodels::common {

     template<class Tscal>
     struct RenderConfig;
     
     template<class Tscal>
        struct RenderConfig {
        //Tscal hfact;
        Tscal gpart_mass;
        unsigned int tree_reduction_level;

    };

}
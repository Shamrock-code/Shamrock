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
 * @file NeighFaceList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <array>

namespace shammodels::zeus {

    template<class Tvec>
    struct OrientedNeighFaceList {
        shamrock::tree::ObjectCache neigh_info;
        Tvec normal;
    };

    template<class Tvec>
    struct NeighFaceList {
        std::array<OrientedNeighFaceList<Tvec>, 6> faces_lists;

        static constexpr u32 i_xm = 0;
        static constexpr u32 i_xp = 1;
        static constexpr u32 i_ym = 2;
        static constexpr u32 i_yp = 3;
        static constexpr u32 i_zm = 4;
        static constexpr u32 i_zp = 5;

        OrientedNeighFaceList<Tvec> &xm() { return faces_lists[i_xm]; }
        OrientedNeighFaceList<Tvec> &xp() { return faces_lists[i_xp]; }
        OrientedNeighFaceList<Tvec> &ym() { return faces_lists[i_ym]; }
        OrientedNeighFaceList<Tvec> &yp() { return faces_lists[i_yp]; }
        OrientedNeighFaceList<Tvec> &zm() { return faces_lists[i_zm]; }
        OrientedNeighFaceList<Tvec> &zp() { return faces_lists[i_zp]; }
    };

} // namespace shammodels::zeus

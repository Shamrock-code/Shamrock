// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchCoordTransform.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/CoordRangeTransform.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchCoord.hpp"
#include "shamsys/legacy/log.hpp"

namespace shamrock::patch {

    template<class Tcoord>
    class PatchCoordTransform {

        using CoordTransform = shammath::CoordRangeTransform<u64_3, Tcoord>;

        CoordTransform transform;

        public:
        inline PatchCoordTransform(
            shammath::CoordRange<u64_3> patch_range, shammath::CoordRange<Tcoord> obj_range)
            : transform(patch_range, obj_range) {}

        inline shammath::CoordRange<Tcoord> to_obj_coord(shammath::CoordRange<u64_3> p) const {
            return transform.transform(p);
        }

        inline shammath::AABB<Tcoord> to_obj_coord(shammath::AABB<u64_3> p) const {
            auto ret = transform.transform(shammath::CoordRange<u64_3>(p.lower, p.upper));
            return {ret.lower, ret.upper};
        }

        PatchCoord<3> to_patch_coord(shammath::CoordRange<Tcoord> obj) const {
            return transform.reverse_transform(obj);
        }

        inline shammath::CoordRange<Tcoord> to_obj_coord(Patch p) const {
            return to_obj_coord(p.get_patch_range());
        }

        inline void print_transform() const { transform.print_transform(); }
    };
} // namespace shamrock::patch

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchCoord.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shammath/CoordRange.hpp"

namespace shamrock::patch {

    template<u32 dim = 3U>
    class PatchCoord {
        public:
        static constexpr u32 splts_count = 1U << dim;

        std::array<u64, dim> coord_min;
        std::array<u64, dim> coord_max;

        PatchCoord() = default;

        PatchCoord(std::array<u64, dim> coord_min, std::array<u64, dim> coord_max)
            : coord_min(coord_min), coord_max(coord_max) {}

        [[nodiscard]] inline static auto get_split_coord(
            std::array<u64, dim> coord_min, std::array<u64, dim> coord_max)
            -> std::array<u64, dim> {
            return {
                (((coord_max[0] - coord_min[0]) + 1) / 2) - 1 + coord_min[0],
                (((coord_max[1] - coord_min[1]) + 1) / 2) - 1 + coord_min[1],
                (((coord_max[2] - coord_min[2]) + 1) / 2) - 1 + coord_min[2]};
        }

        inline static auto get_split(std::array<u64, dim> coord_min, std::array<u64, dim> coord_max)
            -> std::array<PatchCoord, splts_count> {

            std::array<PatchCoord, splts_count> pret;

            auto splts = get_split_coord(coord_min, coord_max);

            u64 split_x = splts[0];
            u64 split_y = splts[1];
            u64 split_z = splts[2];

            pret[0].coord_min[0] = coord_min[0];
            pret[0].coord_min[1] = coord_min[1];
            pret[0].coord_min[2] = coord_min[2];
            pret[0].coord_max[0] = split_x;
            pret[0].coord_max[1] = split_y;
            pret[0].coord_max[2] = split_z;

            pret[1].coord_min[0] = coord_min[0];
            pret[1].coord_min[1] = coord_min[1];
            pret[1].coord_min[2] = split_z + 1;
            pret[1].coord_max[0] = split_x;
            pret[1].coord_max[1] = split_y;
            pret[1].coord_max[2] = coord_max[2];

            pret[2].coord_min[0] = coord_min[0];
            pret[2].coord_min[1] = split_y + 1;
            pret[2].coord_min[2] = coord_min[2];
            pret[2].coord_max[0] = split_x;
            pret[2].coord_max[1] = coord_max[1];
            pret[2].coord_max[2] = split_z;

            pret[3].coord_min[0] = coord_min[0];
            pret[3].coord_min[1] = split_y + 1;
            pret[3].coord_min[2] = split_z + 1;
            pret[3].coord_max[0] = split_x;
            pret[3].coord_max[1] = coord_max[1];
            pret[3].coord_max[2] = coord_max[2];

            pret[4].coord_min[0] = split_x + 1;
            pret[4].coord_min[1] = coord_min[1];
            pret[4].coord_min[2] = coord_min[2];
            pret[4].coord_max[0] = coord_max[0];
            pret[4].coord_max[1] = split_y;
            pret[4].coord_max[2] = split_z;

            pret[5].coord_min[0] = split_x + 1;
            pret[5].coord_min[1] = coord_min[1];
            pret[5].coord_min[2] = split_z + 1;
            pret[5].coord_max[0] = coord_max[0];
            pret[5].coord_max[1] = split_y;
            pret[5].coord_max[2] = coord_max[2];

            pret[6].coord_min[0] = split_x + 1;
            pret[6].coord_min[1] = split_y + 1;
            pret[6].coord_min[2] = coord_min[2];
            pret[6].coord_max[0] = coord_max[0];
            pret[6].coord_max[1] = coord_max[1];
            pret[6].coord_max[2] = split_z;

            pret[7].coord_min[0] = split_x + 1;
            pret[7].coord_min[1] = split_y + 1;
            pret[7].coord_min[2] = split_z + 1;
            pret[7].coord_max[0] = coord_max[0];
            pret[7].coord_max[1] = coord_max[1];
            pret[7].coord_max[2] = coord_max[2];

            return pret;
        }

        inline auto split() -> std::array<PatchCoord, splts_count> {
            return get_split(coord_min, coord_max);
        }

        inline static PatchCoord merge(PatchCoord c1, PatchCoord c2) {
            return PatchCoord(
                {sycl::min(c1.coord_min[0], c2.coord_min[0]),
                 sycl::min(c1.coord_min[1], c2.coord_min[1]),
                 sycl::min(c1.coord_min[2], c2.coord_min[2])},
                {sycl::max(c1.coord_max[0], c2.coord_max[0]),
                 sycl::max(c1.coord_max[1], c2.coord_max[1]),
                 sycl::max(c1.coord_max[2], c2.coord_max[2])});
        }

        inline static PatchCoord merge(std::array<PatchCoord, splts_count> others) {
            return merge(
                merge(merge(others[0], others[1]), merge(others[2], others[3])),
                merge(merge(others[4], others[5]), merge(others[6], others[7])));
        }

        [[nodiscard]] shammath::CoordRange<u64_3> get_patch_range() const {
            return {
                u64_3{coord_min[0], coord_min[1], coord_min[2]},
                u64_3{coord_max[0], coord_max[1], coord_max[2]} + 1};
        }

        template<class T>
        inline static std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>> convert_coord(
            std::array<u64, dim> coord_min,
            std::array<u64, dim> coord_max,
            std::array<u64, dim> pcoord_offset,
            sycl::vec<T, 3> divfact,
            sycl::vec<T, 3> offset) {

            using vec = sycl::vec<T, 3>;

            vec min_bound = vec{coord_min[0] - pcoord_offset[0],
                                coord_min[1] - pcoord_offset[1],
                                coord_min[2] - pcoord_offset[2]}
                                / divfact
                            + offset;
            vec max_bound = (vec{coord_max[0] - pcoord_offset[0],
                                 coord_max[1] - pcoord_offset[1],
                                 coord_max[2] - pcoord_offset[2]}
                             + 1)
                                / divfact
                            + offset;

            return {min_bound, max_bound};
        }
    };

    template<u32 dim = 3U>
    inline bool operator==(const PatchCoord<dim> &lhs, const PatchCoord<dim> &rhs) {
        return (lhs.coord_min == rhs.coord_min) && (lhs.coord_max == rhs.coord_max);
    }
} // namespace shamrock::patch

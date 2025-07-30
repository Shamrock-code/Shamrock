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
 * @file AMRCell.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"

namespace shamrock::amr {

    template<class Tcoord, u32 dim>
    class AMRBlockCoord {

        public:
        static constexpr u32 splts_count = 1U << dim;
        Tcoord bmin, bmax;

        [[nodiscard]] inline static auto get_split_coord(Tcoord bmin, Tcoord bmax) -> Tcoord {
            return (bmax - bmin) / 2 + bmin;
        }

        inline static auto
        get_split(Tcoord bmin, Tcoord bmax) -> std::array<AMRBlockCoord, splts_count> {

            std::array<AMRBlockCoord, splts_count> ret;

            Tcoord splts              = get_split_coord(bmin, bmax);
            std::array<Tcoord, 3> szs = {bmin, splts, bmax};

            auto get_coord = [](u32 i) -> std::array<u32, dim> {
                constexpr u32 NsideBlockPow = 1;
                constexpr u32 Nside         = 1U << NsideBlockPow;
                constexpr u32 side_size     = Nside;
                constexpr u32 block_size    = shambase::pow_constexpr<dim>(Nside);

                if constexpr (dim == 3) {
                    const u32 tmp = i >> NsideBlockPow;
                    return {i % Nside, (tmp) % Nside, (tmp) >> NsideBlockPow};
                }
            };

            for (u32 i = 0; i < splts_count; i++) {
                auto [lx, ly, lz] = get_coord(i);

                // is this the correct order for the refinement ???
                ret[i].bmin = Tcoord{szs[lx].x(), szs[ly].y(), szs[lz].z()};
                ret[i].bmax = Tcoord{szs[lx + 1].x(), szs[ly + 1].y(), szs[lz + 1].z()};
            }

            return ret;
        }

        inline auto split() { return AMRBlockCoord::get_split(bmin, bmax); }

        inline static AMRBlockCoord get_merge(AMRBlockCoord c1, AMRBlockCoord c2) {
            return AMRBlockCoord{sycl::min(c1.bmin, c2.bmin), sycl::max(c1.bmax, c2.bmax)};
        }

        inline static AMRBlockCoord get_merge(std::array<AMRBlockCoord, splts_count> others) {
            return AMRBlockCoord::get_merge(
                AMRBlockCoord::get_merge(
                    AMRBlockCoord::get_merge(others[0], others[1]),
                    AMRBlockCoord::get_merge(others[2], others[3])),
                AMRBlockCoord::get_merge(
                    AMRBlockCoord::get_merge(others[4], others[5]),
                    AMRBlockCoord::get_merge(others[6], others[7])));
        }

        inline static bool are_mergeable(std::array<AMRBlockCoord, splts_count> others) {

            AMRBlockCoord merged = AMRBlockCoord::get_merge(others);

            std::array<AMRBlockCoord, splts_count> splitted = merged.split();

            bool are_same = true;

            static_assert(dim == 3, "only dim 3 is handled");

            if constexpr (dim == 3) {
                for (u32 i = 0; i < splts_count; i++) {
                    are_same = are_same && sham::equals(others[i].bmin, splitted[i].bmin)
                               && sham::equals(others[i].bmax, splitted[i].bmax);
                }
            }

            return are_same;
        }
    };

} // namespace shamrock::amr

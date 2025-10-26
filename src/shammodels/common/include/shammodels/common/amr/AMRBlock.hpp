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
 * @file AMRBlock.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief utility to manipulate AMR blocks
 */

#include "shambase/integer.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <array>

namespace shammodels::amr {

    /**
     * @brief utility class to handle AMR blocks
     *
     * @tparam Tvec
     * @tparam NsideBlockPow
     */
    template<class Tvec, class TgridVec, u32 _NsideBlockPow>
    struct AMRBlock {
        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr u32 dim = shambase::VectorProperties<TgridVec>::dimension;

        static constexpr u32 NsideBlockPow = _NsideBlockPow;
        static constexpr u32 Nside         = 1U << NsideBlockPow;
        static constexpr u32 side_size     = Nside;

        static constexpr u32 block_size = shambase::pow_constexpr<dim>(Nside);

        /**
         * @brief Get the local index within the AMR block
         *
         * @param coord wanted integer coordinates
         * @return constexpr u32 the index
         */
        inline static constexpr u32 get_index(std::array<u32, dim> coord) noexcept {
            static_assert(dim < 5, "not implemented above dim 4");

            if constexpr (dim == 1) {
                return coord[0];
            }

            if constexpr (dim == 2) {
                return coord[0] + Nside * coord[1];
            }

            if constexpr (dim == 3) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
            }

            if constexpr (dim == 4) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2]
                       + Nside * Nside * Nside * coord[3];
            }

            return {};
        }

        inline static constexpr i32 get_index_relative(std::array<i32, dim> coord) noexcept {
            static_assert(dim < 5, "not implemented above dim 4");

            if constexpr (dim == 1) {
                return coord[0];
            }

            if constexpr (dim == 2) {
                return coord[0] + Nside * coord[1];
            }

            if constexpr (dim == 3) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
            }

            if constexpr (dim == 4) {
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2]
                       + Nside * Nside * Nside * coord[3];
            }

            return {};
        }

        inline static constexpr std::array<u32, dim> get_coord(u32 i) noexcept {
            static_assert(dim == 3, "only in dim 3 for now");

            if constexpr (dim == 3) {
                const u32 tmp = i >> NsideBlockPow;
                return {i % Nside, (tmp) % Nside, (tmp) >> NsideBlockPow};
            }

            return {};
        }

        inline static std::pair<Tvec, Tvec> utils_get_cell_coords(
            std::pair<TgridVec, TgridVec> input, u32 lid) {
            Tvec block_min  = input.first.template convert<Tscal>();
            Tvec block_max  = input.second.template convert<Tscal>();
            Tvec delta_cell = (block_max - block_min) / side_size;

            std::array<u32, dim> l_coord = get_coord(lid);

            Tvec cell_offset = Tvec{
                delta_cell.x() * l_coord[0],
                delta_cell.y() * l_coord[1],
                delta_cell.z() * l_coord[2]};

            return {block_min + cell_offset, block_min + cell_offset + delta_cell};
        }

        template<class Func>
        inline static void for_each_cell_in_block(Tvec delta_cell, Func &&functor) noexcept {
            static_assert(dim == 3, "implemented only in dim 3");
            for (u32 ix = 0; ix < side_size; ix++) {
                for (u32 iy = 0; iy < side_size; iy++) {
                    for (u32 iz = 0; iz < side_size; iz++) {
                        u32 i          = get_index({ix, iy, iz});
                        Tvec delta_val = delta_cell * Tvec{ix, iy, iz};
                        functor(i, delta_val);
                    }
                }
            }
        }

        /**
         * @brief for each cell routine for amr block.
         * This handle a loop over all the cells in blocks.
         *
         * \code{.cpp}
         *    Block::for_each_cells(cgh,  mpdat.total_elements,"compite Pis",
         *        [=](u32 block_id, u32 cell_gid){
         *            vec d_cell =
         *                (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
         *                coord_conv_fact;
         *
         *            Tscal rho_i_j_k   = rho[cell_gid];
         *        }
         *    );
         * \endcode
         *
         * @tparam Func
         * @param cgh
         * @param name
         * @param block_cnt
         * @param f
         */
        template<class Func>
        inline static void for_each_cells(
            sycl::handler &cgh, u32 block_cnt, const char *name, Func &&f) {
            // we use one thread per subcell because :
            // double load are avoided because of contiguous L2 cache hit
            // and CF perf opti for GPU, finer threading lead to better latency hidding
            shambase::parallel_for(cgh, block_cnt * block_size, name, [=](u64 id_cell) {
                u32 block_id = id_cell / block_size;
                f(block_id, id_cell);
            });
        }

        template<class Func>
        inline static void for_each_cells_lid(
            sycl::handler &cgh, u32 block_cnt, const char *name, Func &&f) {
            // we use one thread per subcell because :
            // double load are avoided because of contiguous L2 cache hit
            // and CF perf opti for GPU, finer threading lead to better latency hidding
            shambase::parallel_for(cgh, block_cnt * block_size, name, [=](u64 id_cell) {
                u32 block_id = id_cell / block_size;
                u32 lid      = id_cell % block_size;
                f(block_id, lid);
            });
        }

        template<class Acccellcoord>
        inline void for_each_neigh_faces(
            shamrock::tree::ObjectCacheIterator &block_faces_iter,
            Acccellcoord acc_block_min,
            Acccellcoord acc_block_max,
            const u32 id_a,
            const shammath::AABB<TgridVec> aabb_cell) {

            block_faces_iter.for_each_object(id_a, [&](u32 id_b) {
                const Tvec block_b_min = acc_block_min[id_a];
                Tvec block_b_max       = acc_block_max[id_a];
                Tvec delta_cell        = (block_b_max - block_b_min) / side_size;
            });
        }
    };

} // namespace shammodels::amr

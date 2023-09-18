// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRBlock.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief utility to manipulate AMR blocks
 */

#include "shambase/integer.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/type_aliases.hpp"
#include "shammath/AABB.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include <array>

namespace shammodels::amr {

    /**
     * @brief utility class to handle AMR blocks
     *
     * @tparam Tvec
     * @tparam NsideBlockPow
     */
    template<class Tvec, class TgridVec, u32 NsideBlockPow>
    struct AMRBlock {

        static constexpr u32 dim = shambase::VectorProperties<TgridVec>::dimension;

        static constexpr u32 Nside = 1U << NsideBlockPow;
        static constexpr u32 side_size = Nside;

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
                return coord[0] + Nside * coord[1] + Nside * Nside * coord[2] +
                       Nside * Nside * Nside * coord[3];
            }

            return {};
        }

        static constexpr std::array<u32,dim> get_coord(u32 i) noexcept{
            static_assert(dim == 3, "only in dim 3 for now");

            if constexpr (dim == 3) {
                const u32 tmp  = i >> NsideBlockPow;
                return {(tmp) >> NsideBlockPow, (tmp)%Nside,i % Nside};
            }

            return {};
        }

        template<class Func>
        inline static void
        for_each_cell_in_block(Tvec delta_cell, Func && functor) noexcept {
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
        inline static void for_each_cells( sycl::handler &cgh,
                                            u32 block_cnt,
                                            const char * name, Func && f) {
            // we use one thread per subcell because :
            // double load are avoided because of contiguous L2 cache hit
            // and CF perf opti for GPU, finer threading lead to better latency hidding
            shambase::parralel_for(cgh, block_cnt * block_size, name, [=](u64 id_cell) {
                u32 block_id = id_cell / block_size;
                f(block_id, id_cell);
            });
        }

        template<class Func>
        inline static void for_each_cells_lid( sycl::handler &cgh,
                                            u32 block_cnt,
                                            const char * name,
                                             Func && f) {
            // we use one thread per subcell because :
            // double load are avoided because of contiguous L2 cache hit
            // and CF perf opti for GPU, finer threading lead to better latency hidding
            shambase::parralel_for(cgh, block_cnt * block_size, name, [=](u64 id_cell) {
                u32 block_id = id_cell / block_size;
                u32 lid = id_cell % block_size;
                f(block_id, lid);
            });
        }






        template<class Acccellcoord>
        inline void for_each_neigh_faces(
            shamrock::tree::ObjectCacheIterator & block_faces_iter, 
            Acccellcoord acc_block_min,
            Acccellcoord acc_block_max,
            const u32 id_a,
            const shammath::AABB<TgridVec> aabb_cell){

            block_faces_iter.for_each_object(id_a, [&](u32 id_b) {
                
                const Tvec block_b_min = acc_block_min[id_a];
                Tvec block_b_max = acc_block_max[id_a];
                Tvec delta_cell = (block_b_max - block_b_min)/side_size;



            });

        }
    };
    
    

} // namespace shammodels::amr
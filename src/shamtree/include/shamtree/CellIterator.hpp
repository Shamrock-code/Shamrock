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
 * @file CellIterator.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <utility>

namespace shamtree {

    struct CellIteratorAccessed {
        const u32 *sort_index_map;  ///< Pointer to the sort index map
        const u32 *reduc_index_map; ///< Pointer to the reduction index map

        template<class Functor_iter>
        inline void for_each_in_cell_range(
            const u32 &cell_id_begin, const u32 &cell_id_end, Functor_iter &&func_it) const {
            // loop on particle indexes
            uint min_ids = reduc_index_map[cell_id_begin];
            uint max_ids = reduc_index_map[cell_id_end];

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                // recover old index before morton sort
                uint id_b = sort_index_map[id_s];

                // iteration function
                func_it(id_b);
            }
        }

        /**
         * @brief Iterate over all particles in a given cell.
         *
         * @param[in] cell_id ID of the cell to iterate over.
         * @param[in] func_it function to call for each particle in the cell.
         *
         * This function takes a cell ID and a functor as argument. It will then
         * iterate over all particles in the given cell and call the functor
         * with each particle's index as argument.
         */
        template<class Functor_iter>
        inline void for_each_in_leaf_cell(const u32 &cell_id, Functor_iter &&func_it) const {
            for_each_in_cell_range(cell_id, cell_id + 1, std::forward<Functor_iter>(func_it));
        }
    };

    /**
     * @class CellIterator
     * @brief Iterator over cells of a BinaryTree.
     */
    struct CellIterator {
        sham::DeviceBuffer<u32> &buf_sort_index_map;  ///< Sort index map buffer
        sham::DeviceBuffer<u32> &buf_reduc_index_map; ///< Reduction index map buffer

        using acc = CellIteratorAccessed;

        /**
         * @brief Get a read-only access to the buffers.
         *
         * @param[in] deps Event list to register the access in.
         * @return A structure containing const pointers to the buffers.
         *
         * This function returns a struct containing const pointers to the
         * `buf_sort_index_map` and `buf_reduc_index_map` buffers. The
         * access is registered in the `deps` event list.
         */
        inline acc get_read_access(sham::EventList &deps) const {
            return acc{
                buf_sort_index_map.get_read_access(deps),
                buf_reduc_index_map.get_read_access(deps)};
        }

        /**
         * @brief Completes the event state for the associated buffers.
         *
         * @param[in] e The SYCL event to register for the buffers.
         *
         * This function registers the provided SYCL event in the event state
         * of both `buf_sort_index_map` and `buf_reduc_index_map`, indicating
         * that the event has been completed for these buffers.
         */
        inline void complete_event_state(sycl::event e) const {
            buf_sort_index_map.complete_event_state(e);
            buf_reduc_index_map.complete_event_state(e);
        }
    };

    /// host version of the cell iterator
    struct CellIteratorHost {
        std::vector<u32> sort_index_map;  ///< Sort index map
        std::vector<u32> reduc_index_map; ///< Reduction index map

        using acc = CellIteratorAccessed;

        /// get read only accessor
        inline acc get_read_access() const {
            return acc{sort_index_map.data(), reduc_index_map.data()};
        }
    };

} // namespace shamtree

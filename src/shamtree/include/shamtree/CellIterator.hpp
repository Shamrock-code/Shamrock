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
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include <utility>

namespace shamtree {

    struct CellIteratorAccessed {
        const u32 *sort_index_map;  ///< Pointer to the sort index map
        const u32 *reduc_index_map; ///< Pointer to the reduction index map
        const u32 *endrange; ///< Id of the other end of the index range corresponding to the cell
        u32 offset_leaf;     ///< number of internal cells & offset to retrieve the first leaf

        /// is the given id a leaf (Note that if there is no internal cell every node is a leaf)
        inline bool is_id_leaf(u32 id) const { return id >= offset_leaf; }

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
        inline void for_each_in_cell(const u32 &cell_id, Functor_iter &&func_it) const {

            bool is_leaf = is_id_leaf(cell_id);

            // internal cell id or leaf id (hence the sub internal_cell_count if leaf)
            u32 cbeg = (is_leaf) ? cell_id - offset_leaf : cell_id;

            // other end of the cell range (either ourself if leaf, or the endrange if internal)
            // this exclude the upper bound as the +1 must be made after the reordering
            u32 cend = ((is_leaf) ? cbeg : endrange[cbeg]);

            // tree cell index range
            uint c1 = sham::min(cbeg, cend);
            uint c2 = sham::max(cbeg, cend);

            u32 _startrange = reduc_index_map[c1];
            u32 _endrange   = reduc_index_map[c2 + 1]; // <--- this +1

            for (unsigned int id_s = _startrange; id_s < _endrange; id_s++) {

                // recover old index before morton sort
                uint id_b = sort_index_map[id_s];

                // iteration function
                func_it(id_b);
            }
        }
    };

    /**
     * @class CellIterator
     * @brief Iterator over cells of a BinaryTree.
     */
    struct CellIterator {
        const sham::DeviceBuffer<u32> &buf_sort_index_map;  ///< Sort index map buffer
        const sham::DeviceBuffer<u32> &buf_reduc_index_map; ///< Reduction index map buffer
        const sham::DeviceBuffer<u32> &buf_endrange;        ///< End range buffer
        u32 offset_leaf; ///< number of internal cells & offset to retrieve the first leaf

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
                buf_reduc_index_map.get_read_access(deps),
                buf_endrange.get_read_access(deps),
                offset_leaf};
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
            buf_endrange.complete_event_state(e);
        }
    };

    /// host version of the cell iterator
    struct CellIteratorHost {
        std::vector<u32> sort_index_map;  ///< Sort index map
        std::vector<u32> reduc_index_map; ///< Reduction index map
        std::vector<u32> endrange;        ///< End range
        u32 offset_leaf; ///< number of internal cells & offset to retrieve the first leaf

        using acc = CellIteratorAccessed;

        /// get read only accessor
        inline acc get_read_access() const {
            return acc{sort_index_map.data(), reduc_index_map.data(), endrange.data(), offset_leaf};
        }
    };

} // namespace shamtree

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DiscontinuousIterator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header for PatchData related function and declaration
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shambase/aliases_int.hpp"

#include <bitset>

namespace shammath {
    /**
     * @brief Discontinuous Iterator 
     * will iterate over every values in an integer set in the most discontinuous way possible
     *
     * Usage :
     * \code{.cpp}
     * DynamicIdGenerator<i32, group_size> id_gen(q);
     * q.submit([&](sycl::handler &cgh) {
     *
     *    auto dyn_id = id_gen.get_access(cgh);
     *
     *    cgh.parallel_for(sycl::nd_range<1>{len, group_size},
     *        [=](sycl::nd_item<1> id) {
     *
     *            atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);
     *
     *            u32 group_tile_id = group_id.dyn_group_id;
     *            u32 global_id = group_id.dyn_global_id;
     *
     *            // kernel execution
     *
     *        });
     *
     * });
     * \endcode
     * 
     * @tparam T 
     */
    template<class T>
    class DiscontinuousIterator {
        public:
        constexpr static u32 bitcount = sizeof(T) * 8;

        T offset;

        std::bitset<bitcount> max;
        T tmax;
        std::bitset<bitcount> current;
        int firstbit;
        bool done;

        DiscontinuousIterator(T tmin, T tmax) : offset(tmin), tmax(tmax - tmin), max(tmax - tmin), current(0) {

            done = !(tmin < tmax);

            for (firstbit = bitcount - 1; firstbit >= 0; firstbit--) {
                if (max[firstbit]) {
                    break;
                }
            }
        }

        bool is_done() { return done; }

        T next() {
            T tmp = get();
            advance_it();
            return tmp;
        }

        T get() { return current.to_ullong() + offset; }

        void advance_it() {
            do {
                bool carry  = true; // backward
                int pointer = firstbit;
                while (carry && !done) {
                    if (current[pointer]) {
                        current.flip(pointer);
                        pointer--;
                        if (pointer < 0) {
                            done = true;
                        }
                    } else {
                        carry = false;
                        current.flip(pointer);
                    }
                }
            } while (current.to_ullong() >= tmax);
        }
    };
    
} // namespace shammath
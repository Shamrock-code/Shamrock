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
 * @file dtt_parralel_select.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTParallelSelect {

        using Tscal = shambase::VecComponent<Tvec>;

        inline static bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
            Tvec s_a      = (a.upper - a.lower);
            Tvec s_b      = (b.upper - b.lower);
            Tvec r_a      = (a.upper + a.lower) / 2;
            Tvec r_b      = (b.upper + b.lower) / 2;
            Tvec delta_ab = r_a - r_b;

            Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

            if (delta_ab_sq == 0) {
                return false;
            }

            Tscal s_a_sq = sham::dot(s_a, s_a);
            Tscal s_b_sq = sham::dot(s_b, s_b);

            Tscal r_a_sq = sham::dot(r_a, r_a);
            Tscal r_b_sq = sham::dot(r_b, r_b);

            Tscal theta_sq = (s_a_sq + s_b_sq) / delta_ab_sq;

            return theta_sq < theta_crit * theta_crit;
        }

        inline static shamtree::DTTResult
        dtt(sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit) {

            u32 total_cell_count = bvh.structure.get_total_cell_count();

            sham::DeviceBuffer<u32> count_m2m(total_cell_count + 1, dev_sched);
            sham::DeviceBuffer<u32> count_p2p(total_cell_count + 1, dev_sched);
            count_m2m.set_val_at_idx(0, 0);
            count_p2p.set_val_at_idx(0, 0);

            // count the number of interactions for each cell

            /////////////////////////////////////////////////////////////

            // scans the counts
            sham::DeviceBuffer<u32> scan_m2m
                = shamalgs::numeric::scan_exclusive(dev_sched, count_m2m, total_cell_count);
            sham::DeviceBuffer<u32> scan_p2p
                = shamalgs::numeric::scan_exclusive(dev_sched, count_p2p, total_cell_count);

            // alloc results buffers
            u32 total_count_m2m = scan_m2m.get_val_at_idx(total_cell_count);
            u32 total_count_p2p = scan_p2p.get_val_at_idx(total_cell_count);

            sham::DeviceBuffer<u32_2> idx_m2m(total_count_m2m, dev_sched);
            sham::DeviceBuffer<u32_2> idx_p2p(total_count_p2p, dev_sched);

            // relaunch the previous kernel but write the indexes this time

            /////////////////////////////////////////////////////////////

            using ObjectIterator  = shamtree::CLBVHObjectIterator<Tmorton, Tvec, dim>;
            ObjectIterator obj_it = bvh.get_object_iterator();

            using ObjItAcc = typename ObjectIterator::acc;

            return DTTResult{std::move(idx_m2m), std::move(idx_p2p)};
        }
    };

} // namespace shamtree::details



// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HilbertLoadBalance.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief implementation of the hilbert curve load balancing
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/loadbalance/LoadBalanceStrategy.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

inline void apply_node_patch_packing(
    std::vector<shamrock::patch::Patch> &global_patch_list, std::vector<i32> &new_owner_table) {

    // Note that there seems to be a data race here
    // However this should never happends as packing index will only point toward a patch without
    // packing. As such the data we are accessing should never be modified during this loop.
#pragma omp parallel for
    for (size_t i = 0; i < global_patch_list.size(); i++) {
        if (global_patch_list[i].pack_node_index != u64_max) {
            new_owner_table[i] = new_owner_table[global_patch_list[i].pack_node_index];
        }
    }
}

namespace shamrock::scheduler {

    template<class T>
    class Compute_HilbLoad;
    template<class T>
    class Write_chosen_node;
    template<class T>
    class Edit_chosen_node;

    template<class hilbert_num>
    LoadBalancingChangeList HilbertLoadBalance<hilbert_num>::make_change_list(
        std::vector<shamrock::patch::Patch> &global_patch_list) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        // result
        LoadBalancingChangeList change_list;

        using Torder  = hilbert_num;
        using Tweight = u64;
        using LBTile  = TileWithLoad<Torder, Tweight>;

        // generate hilbert code, load value, and index before sort
        std::vector<LBTile> patch_dt(global_patch_list.size());

#pragma omp parallel for
        for (u64 i = 0; i < global_patch_list.size(); i++) {

            const Patch &p = global_patch_list[i];

            patch_dt[i]
                = {SFC::icoord_to_hilbert(p.coord_min[0], p.coord_min[1], p.coord_min[2]),
                   p.load_value};
        }

        std::vector<i32> new_owner_table = load_balance(std::move(patch_dt));

        // apply patch packing in same node for merge
        apply_node_patch_packing(global_patch_list, new_owner_table);

        // make change list
        {
            std::vector<u64> load_per_node(shamcomm::world_size());

            std::vector<i32> tags_it_node(shamcomm::world_size());
            for (u64 i = 0; i < global_patch_list.size(); i++) {

                i32 old_owner = global_patch_list[i].node_owner_id;
                i32 new_owner = new_owner_table[i];

                // TODO add bool for optional print verbosity
                // std::cout << i << " : " << old_owner << " -> " << new_owner << std::endl;
                if (new_owner != old_owner) {

                    using ChangeOp = LoadBalancingChangeList::ChangeOp;

                    ChangeOp op;
                    op.patch_idx      = i;
                    op.patch_id       = global_patch_list[i].id_patch;
                    op.rank_owner_new = new_owner;
                    op.rank_owner_old = old_owner;
                    op.tag_comm       = tags_it_node[old_owner];

                    change_list.change_ops.push_back(op);
                    tags_it_node[old_owner]++;
                }

                load_per_node[new_owner_table[i]] += global_patch_list[i].load_value;
            }

            // shamlog_debug_ln("HilbertLoadBalance", "loads after balancing");
            f64 min = shambase::VectorProperties<f64>::get_inf();
            f64 max = -shambase::VectorProperties<f64>::get_inf();
            f64 avg = 0;
            f64 var = 0;

            i32 world_size = shamcomm::world_size();

#pragma omp parallel for reduction(min : min) reduction(max : max) reduction(+ : avg)
            for (i32 nid = 0; nid < world_size; nid++) {
                f64 val = load_per_node[nid];
                min     = sycl::fmin(min, val);
                max     = sycl::fmax(max, val);
                avg += val;
            }

            if (shamcomm::world_rank() == 0
                && shamcomm::logs::get_loglevel() >= shamcomm::logs::log_debug) {
                for (i32 nid = 0; nid < world_size; nid++) {
                    shamlog_debug_ln(
                        "HilbertLoadBalance", "node :", nid, "load :", load_per_node[nid]);
                }
            }
            avg /= world_size;

#pragma omp parallel for reduction(+ : var)
            for (i32 nid = 0; nid < world_size; nid++) {
                f64 val = load_per_node[nid];
                var += (val - avg) * (val - avg);
            }
            var /= world_size;

            if (shamcomm::world_rank() == 0) {
                std::string str = "Loadbalance stats : \n";
                str += shambase::format("    npatch = {}\n", global_patch_list.size());
                str += shambase::format("    min = {}\n", min);
                str += shambase::format("    max = {}\n", max);
                str += shambase::format("    avg = {}\n", avg);
                if (max == 0) {
                    str += "    efficiency = ???%";
                } else {
                    str += shambase::format(
                        "    efficiency = {:.2f}%", 100 - (100 * (max - min) / max));
                }
                logger::info_ln("LoadBalance", str);
            }
        }

        return change_list;
    }

    template class HilbertLoadBalance<u64>;
    template class HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

} // namespace shamrock::scheduler

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
 * @file LoadBalanceStrategy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief implementation of the hilbert curve load balancing
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <vector>

namespace shamrock::scheduler {
    template<class Torder, class Tweight>
    struct TileWithLoad {
        Torder ordering_val;
        Tweight load_value;
    };
} // namespace shamrock::scheduler

namespace shamrock::scheduler::details {

    template<class Torder, class Tweight>
    struct LoadBalancedTile {
        Torder ordering_val;
        Tweight load_value;
        Tweight accumulated_load_value;
        u64 index;
        i32 new_owner;

        LoadBalancedTile(TileWithLoad<Torder, Tweight> in, u64 inindex)
            : ordering_val(in.ordering_val), load_value(in.load_value), index(inindex) {}
    };

    template<class Torder, class Tweight>
    inline void apply_ordering(std::vector<LoadBalancedTile<Torder, Tweight>> &lb_vec) {
        using LBTileResult = LoadBalancedTile<Torder, Tweight>;
        std::sort(lb_vec.begin(), lb_vec.end(), [](LBTileResult &left, LBTileResult &right) {
            return left.ordering_val < right.ordering_val;
        });
    }

    template<class Torder, class Tweight>
    inline std::vector<i32> lb_startegy_parralel_sweep(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {

        using LBTile       = TileWithLoad<Torder, Tweight>;
        using LBTileResult = details::LoadBalancedTile<Torder, Tweight>;

        std::vector<LBTileResult> res;
        for (u64 i = 0; i < lb_vector.size(); i++) {
            res.push_back(LBTileResult{lb_vector[i], i});
        }

        // apply the ordering
        apply_ordering(res);

        // compute increments for load
        u64 accum = 0;
        for (LBTileResult &tile : res) {
            u64 cur_val                 = tile.load_value;
            tile.accumulated_load_value = accum;
            accum += cur_val;
        }

        double target_datacnt = double(res[res.size() - 1].accumulated_load_value) / wsize;

        for (LBTileResult &tile : res) {
            tile.new_owner
                = (target_datacnt == 0)
                      ? 0
                      : sycl::clamp(
                            i32(tile.accumulated_load_value / target_datacnt), 0, wsize - 1);
        }

        if (shamcomm::world_rank() == 0) {
            for (LBTileResult t : res) {
                shamlog_debug_ln(
                    "HilbertLoadBalance",
                    t.ordering_val,
                    t.accumulated_load_value,
                    t.index,
                    (target_datacnt == 0)
                        ? 0
                        : sycl::clamp(
                              i32(t.accumulated_load_value / target_datacnt), 0, i32(wsize) - 1),
                    (target_datacnt == 0) ? 0 : (t.accumulated_load_value / target_datacnt));
            }
        }

        std::vector<i32> new_owners(res.size());
        for (LBTileResult &tile : res) {
            new_owners[tile.index] = tile.new_owner;
        }

        return new_owners;
    }

    template<class Torder, class Tweight>
    inline std::vector<i32>
    lb_startegy_roundrobin(const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {

        using LBTile       = TileWithLoad<Torder, Tweight>;
        using LBTileResult = details::LoadBalancedTile<Torder, Tweight>;

        std::vector<LBTileResult> res;
        for (u64 i = 0; i < lb_vector.size(); i++) {
            res.push_back(LBTileResult{lb_vector[i], i});
        }

        // apply the ordering
        apply_ordering(res);

        // compute increments for load
        u64 accum = 0;
        for (LBTileResult &tile : res) {
            tile.accumulated_load_value = accum;
            // modify the lB above by assuming that each patch has the same load
            // which effectivelly does a round robin balancing
            accum += 1;
        }

        double target_datacnt = double(res[res.size() - 1].accumulated_load_value) / wsize;

        for (LBTileResult &tile : res) {
            tile.new_owner
                = (target_datacnt == 0)
                      ? 0
                      : sycl::clamp(
                            i32(tile.accumulated_load_value / target_datacnt), 0, wsize - 1);
        }

        if (shamcomm::world_rank() == 0) {
            for (LBTileResult t : res) {
                shamlog_debug_ln(
                    "HilbertLoadBalance",
                    t.ordering_val,
                    t.accumulated_load_value,
                    t.index,
                    (target_datacnt == 0)
                        ? 0
                        : sycl::clamp(
                              i32(t.accumulated_load_value / target_datacnt), 0, i32(wsize) - 1),
                    (target_datacnt == 0) ? 0 : (t.accumulated_load_value / target_datacnt));
            }
        }

        std::vector<i32> new_owners(res.size());
        for (LBTileResult &tile : res) {
            new_owners[tile.index] = tile.new_owner;
        }

        return new_owners;
    }

    struct LBMetric {
        f64 min;
        f64 max;
        f64 mean;
        f64 stddev;
    };

    template<class Torder, class Tweight>
    inline LBMetric compute_LB_metric(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector,
        const std::vector<i32> &new_owners,
        i32 world_size) {

        std::vector<u64> load_per_node(world_size, 0);

        for (u64 i = 0; i < lb_vector.size(); i++) {
            load_per_node[new_owners[i]] += lb_vector[i].load_value;
        }

        f64 min = shambase::VectorProperties<f64>::get_inf();
        f64 max = -shambase::VectorProperties<f64>::get_inf();
        f64 avg = 0;
        f64 var = 0;

        for (i32 nid = 0; nid < world_size; nid++) {
            f64 val = load_per_node[nid];
            min     = sycl::fmin(min, val);
            max     = sycl::fmax(max, val);
            avg += val;

            // shamlog_debug_ln("HilbertLoadBalance", "node :",nid, "load :",load_per_node[nid]);
        }
        avg /= world_size;
        for (i32 nid = 0; nid < world_size; nid++) {
            f64 val = load_per_node[nid];
            var += (val - avg) * (val - avg);
        }
        var /= world_size;

        return {min, max, avg, sycl::sqrt(var)};
    }

} // namespace shamrock::scheduler::details

namespace shamrock::scheduler {

    /**
     * @brief load balance the input vector
     *
     * @tparam Torder ordering value (hilbert, morton, ...)
     * @tparam Tweight weight type
     * @param lb_vector
     * @return std::vector<i32> The new owner list
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> load_balance(
        std::vector<TileWithLoad<Torder, Tweight>> &&lb_vector,
        i32 world_size = shamcomm::world_size()) {

        auto tmpres        = details::lb_startegy_parralel_sweep(lb_vector, world_size);
        auto metric_psweep = details::compute_LB_metric(lb_vector, tmpres, world_size);

        auto tmpres_2      = details::lb_startegy_roundrobin(lb_vector, world_size);
        auto metric_rrobin = details::compute_LB_metric(lb_vector, tmpres_2, world_size);

        if (metric_rrobin.max < metric_psweep.max) {
            tmpres = tmpres_2;
        }

        if (shamcomm::world_rank() == 0) {
            logger::info_ln("LoadBalance", "summary :");
            logger::info_ln(
                "LoadBalance",
                " - strategy \"psweep\" : max =",
                metric_psweep.max,
                "min =",
                metric_psweep.min);
            logger::info_ln(
                "LoadBalance",
                " - strategy \"round robin\" : max =",
                metric_rrobin.max,
                "min =",
                metric_rrobin.min);
        }
        return tmpres;
    }

} // namespace shamrock::scheduler

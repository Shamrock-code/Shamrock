// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HilbertLoadBalance.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of the hilbert curve load balancing
 *
 */

#include "HilbertLoadBalance.hpp"

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

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

        LoadBalancingChangeList change_list;

        // generate hilbert code, load value, and index before sort

        // std::tuple<hilbert code ,load value ,index in global_patch_list>
        std::vector<std::tuple<hilbert_num, u64, u64>> patch_dt(global_patch_list.size());

        {

            sycl::buffer<std::tuple<hilbert_num, u64, u64>> dt_buf(patch_dt.data(),
                                                                   patch_dt.size());
            sycl::buffer<Patch> patch_buf(global_patch_list.data(), global_patch_list.size());

            for (u64 i = 0; i < global_patch_list.size(); i++) {

                Patch p = global_patch_list[i];

                patch_dt[i] = {
                    SFC::icoord_to_hilbert(p.coord_min[0], p.coord_min[1], p.coord_min[2]),
                    p.load_value,
                    i};
            }
        }

        // sort hilbert code
        std::sort(patch_dt.begin(), patch_dt.end());

        u64 accum = 0;
        // compute increments for load
        for (u64 i = 0; i < global_patch_list.size(); i++) {
            u64 cur_val              = std::get<1>(patch_dt[i]);
            std::get<1>(patch_dt[i]) = accum;
            accum += cur_val;
        }

        //*
        {
            double target_datacnt = double(std::get<1>(patch_dt[global_patch_list.size() - 1])) /
                                    shamcomm::world_size();
            for (auto t : patch_dt) {
                logger::debug_ln("HilbertLoadBalance",
                                 std::get<0>(t),
                                 std::get<1>(t),
                                 std::get<2>(t),
                                 sycl::clamp(i32(std::get<1>(t) / target_datacnt),
                                             0,
                                             i32(shamcomm::world_size()) - 1),
                                 (std::get<1>(t) / target_datacnt));
            }
        }
        //*/

        // compute new owners
        std::vector<i32> new_owner_table(global_patch_list.size());
        {

            sycl::buffer<std::tuple<hilbert_num, u64, u64>> dt_buf(patch_dt.data(),
                                                                   patch_dt.size());
            sycl::buffer<i32> new_owner(new_owner_table.data(), new_owner_table.size());
            sycl::buffer<Patch> patch_buf(global_patch_list.data(), global_patch_list.size());

            sycl::range<1> range{global_patch_list.size()};

            shamsys::instance::get_alt_queue().submit([&](sycl::handler &cgh) {
                auto pdt         = dt_buf.template get_access<sycl::access::mode::read>(cgh);
                auto chosen_node = new_owner.get_access<sycl::access::mode::discard_write>(cgh);

                // TODO [potential issue] here must check that the conversion to double doesn't mess
                // up the target dt_cnt or find another way
                double target_datacnt =
                    double(std::get<1>(patch_dt[global_patch_list.size() - 1])) /
                    shamcomm::world_size();

                i32 wsize = shamcomm::world_size();

                cgh.parallel_for<Write_chosen_node<hilbert_num>>(range, [=](sycl::item<1> item) {
                    u64 i = (u64)item.get_id(0);

                    u64 id_ptable = std::get<2>(pdt[i]);

                    chosen_node[id_ptable] =
                        sycl::clamp(i32(std::get<1>(pdt[i]) / target_datacnt), 0, wsize - 1);
                });
            });

            // pack nodes
            shamsys::instance::get_alt_queue().submit([&](sycl::handler &cgh) {
                auto ptch = patch_buf.get_access<sycl::access::mode::read>(cgh);
                // auto pdt  = dt_buf.get_access<sycl::access::mode::read>(cgh);
                auto chosen_node = new_owner.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<Edit_chosen_node<hilbert_num>>(range, [=](sycl::item<1> item) {
                    u64 i = (u64)item.get_id(0);

                    if (ptch[i].pack_node_index != u64_max) {
                        chosen_node[i] = chosen_node[ptch[i].pack_node_index];
                    }
                });
            });
        }

        // make change list
        {
            std::vector<u64> load_per_node(shamcomm::world_size());

            std::vector<i32> tags_it_node(shamcomm::world_size());
            for (u64 i = 0; i < global_patch_list.size(); i++) {

                i32 old_owner = global_patch_list[i].node_owner_id;
                i32 new_owner = new_owner_table[i];

                // TODO add bool for optional print verbosity
                // std::cout << i << " : " << old_owner << " -> " << new_owner << std::endl;

                using ChangeOp = LoadBalancingChangeList::ChangeOp;

                ChangeOp op;
                op.patch_idx      = i;
                op.patch_id       = global_patch_list[i].id_patch;
                op.rank_owner_new = new_owner;
                op.rank_owner_old = old_owner;
                op.tag_comm       = tags_it_node[old_owner];

                if (new_owner != old_owner) {
                    change_list.change_ops.push_back(op);
                    tags_it_node[old_owner]++;
                }

                load_per_node[new_owner_table[i]] += global_patch_list[i].load_value;
            }

            //logger::debug_ln("HilbertLoadBalance", "loads after balancing");
            f64 min = shambase::VectorProperties<f64>::get_inf();
            f64 max = -shambase::VectorProperties<f64>::get_inf();
            f64 avg = 0;
            f64 var = 0;

            for (i32 nid = 0; nid < shamcomm::world_size(); nid++) {
                f64 val = load_per_node[nid];
                min = sycl::fmin(min, val);
                max = sycl::fmax(max, val);
                avg += val;
                
                logger::debug_ln("HilbertLoadBalance", nid, load_per_node[nid]);
            }
            avg /= shamcomm::world_size();
            for (i32 nid = 0; nid < shamcomm::world_size(); nid++) {
                f64 val = load_per_node[nid];
                var += (val - avg)*(val - avg);
            }
            var /= shamcomm::world_size();

            if(shamcomm::world_rank() == 0){
                std::string str = "Loadbalance stats : \n";
                str += shambase::format("    npatch = {}\n", global_patch_list.size() );
                str += shambase::format("    min = {}\n", min);
                str += shambase::format("    max = {}\n", max);
                str += shambase::format("    avg = {}\n", avg);
                str += shambase::format("    efficiency = {:.2f}%", 100 - (100*(max - min)/max));
                logger::info_ln("LoadBalance", str);
            }
        }

        return change_list;
    }

    template class HilbertLoadBalance<u64>;
    template class HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

} // namespace shamrock::scheduler
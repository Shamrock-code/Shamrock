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
 * @file interface_handler_impl_tree.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "interf_impl_util.hpp"
#include "interface_handler_impl_list.hpp"
#include "shamrock/legacy/comm/sparse_communicator.hpp"
#include "shamrock/legacy/patch/utility/compute_field.hpp"
#include "shamrock/legacy/utils/interact_crit_utils.hpp"
#include "shamtree/RadixTree.hpp"
//%Impl status : Clean unfinished

template<class pos_prec, class u_morton>
class Interfacehandler<Tree_Send, pos_prec, RadixTree<u_morton, sycl::vec<pos_prec, 3>>> {

    public:
    using flt = pos_prec;
    using vec = sycl::vec<flt, 3>;

    private:
    using RadixTreePtr = std::unique_ptr<RadixTree<u_morton, vec>>;
    using CutTree      = typename RadixTree<u_morton, vec>::CuttedTree;

    // Store the result of a tree cut
    struct CommListingSend {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        vec applied_offset;
        i32_3 periodicity_vector;

        vec receiver_box_min;
        vec receiver_box_max;
    };

    struct UnrolledCutTree {
        std::vector<std::unique_ptr<RadixTree<u_morton, vec>>> list_rtree;
        std::vector<std::unique_ptr<sycl::buffer<u32>>> list_new_node_id_to_old;
        std::vector<std::unique_ptr<sycl::buffer<u32>>> list_pdat_extract_id;
    };

    // contain the list of interface that this node should send
    std::vector<CommListingSend> interf_send_map;
    UnrolledCutTree tree_send_map;

    template<class InteractCrit, class... Args>
    inline void internal_compute_interf_list(
        PatchScheduler &sched,
        SerialPatchTree<vec> &sptree,
        SimulationDomain<flt> &bc,
        std::unordered_map<u64, RadixTreePtr> &rtrees,
        const InteractCrit &interact_crit,
        Args... args);

    template<class T>
    struct field_extract_type {
        using type = void;
    };
    template<class T>
    struct field_extract_type<legacy::PatchField<T>> {
        using type = T;
    };

    std::unique_ptr<SparsePatchCommunicator> communicator;

    public:
    template<class InteractCrit, class... Args>
    struct check {
        static constexpr bool has_patch_special_case
            = (std::is_same<
                decltype(InteractCrit::interact_cd_cell_patch),
                bool(
                    vec, vec, vec, vec, field_extract_type<Args>..., field_extract_type<Args>...)>::
                   value);
        static_assert(
            has_patch_special_case,
            "malformed call type should be bool(vec,vec,vec,vec,(types of the inputs "
            "field)...(types of the inputs field)...)");
    };

    // for now interact crit has shape (vec,vec) -> bool
    // in order to pass for exemple h max we need a full tree field (patch field + radix tree field)
    template<class InteractCrit, class... Args>
    inline void compute_interface_list(
        PatchScheduler &sched,
        SerialPatchTree<vec> &sptree,
        SimulationDomain<flt> &bc,
        std::unordered_map<u64, RadixTreePtr> &rtrees,
        InteractCrit &&interact_crit,
        Args &...args) {

        // check<InteractCrit,Args...>{};
        // constexpr bool has_patch_special_case =
        // (std::is_same<decltype(InteractCrit::interact_cd_cell_patch),bool(field_extract_type<decltype(args)>...)>::value);
        // static_assert(has_patch_special_case, "special case must be written for this");//TODO
        // better err msg

        internal_compute_interf_list(
            sched, sptree, bc, rtrees, interact_crit, args.get_buffers()...);
    }

    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    void initial_fetch(PatchScheduler &sched) {

        std::vector<u64_2> send_vec;

        for (auto &comm : interf_send_map) {
            u64 sender_idx = sched.patch_list.id_patch_to_global_idx[comm.sender_patch_id];
            u64 recv_idx   = comm.global_patch_idx_recv;
            send_vec.push_back(u64_2{sender_idx, recv_idx});
        }

        communicator = std::make_unique<SparsePatchCommunicator>(
            sched.patch_list.global, std::move(send_vec));
        communicator->fetch_comm_table();

        shamlog_debug_ln("Interfaces", "fetching comm table"); // TODO Add bandwidth check
    }

    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    SparseCommResult<RadixTree<u_morton, vec>> tree_recv_map;
    void comm_trees() { tree_recv_map = communicator->sparse_exchange(tree_send_map.list_rtree); }

    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    SparseCommResult<shamrock::patch::PatchData> comm_pdat(PatchScheduler &sched) {

        using namespace shamrock::patch;

        SparseCommSource<PatchData> src;

        for (u32 i = 0; i < interf_send_map.size(); i++) {
            auto &comm             = interf_send_map[i];
            UnrolledCutTree &ctree = tree_send_map;

            PatchData &pdat_to_cut = sched.patch_data.get_pdat(comm.sender_patch_id);

            src.push_back(std::make_unique<PatchData>(sched.pdl));

            pdat_to_cut.append_subset_to(
                *ctree.list_pdat_extract_id[i],
                ctree.list_pdat_extract_id[i]->size(),
                *src[src.size() - 1]);
        }

        return communicator->sparse_exchange(src);
    }

    template<class T>
    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    SparseCommResult<RadixTreeField<T>> comm_tree_field(
        PatchScheduler &sched,
        std::unordered_map<u64, std::unique_ptr<RadixTreeField<T>>> &tree_fields) {

        SparseCommSource<RadixTreeField<T>> src;

        for (u32 i = 0; i < interf_send_map.size(); i++) {
            CommListingSend &comm  = interf_send_map[i];
            UnrolledCutTree &ctree = tree_send_map;

            std::unique_ptr<RadixTreeField<T>> &rtree_field_src = tree_fields[comm.sender_patch_id];

            src.push_back(std::make_unique<RadixTreeField<T>>(
                *rtree_field_src, *ctree.list_new_node_id_to_old[i]));
        }

        return communicator->sparse_exchange(src);
    }

    template<class Function>
    void for_each_interface(u64 patch_id, Function &&fct);
};

template<class pos_prec, class u_morton>
template<class InteractCrit, class... Args>
void Interfacehandler<Tree_Send, pos_prec, RadixTree<u_morton, sycl::vec<pos_prec, 3>>>::
    internal_compute_interf_list(
        PatchScheduler &sched,
        SerialPatchTree<vec> &sptree,
        SimulationDomain<flt> &bc,
        std::unordered_map<u64, RadixTreePtr> &rtrees,
        const InteractCrit &interact_crit,
        Args... args) {

    const vec per_vec = bc.get_periodicity_vector();

    const u64 local_pcount  = sched.patch_list.local.size();
    const u64 global_pcount = sched.patch_list.global.size();

    shamlog_debug_ln("Interfacehandler", "computing interface list");

    impl::generator::GeneratorBuffer<flt> gen{sched};

    auto append_interface = [&](i32_3 periodicity_vec) -> auto {
        using namespace impl;

        // meaning in the interface we look at r |-> r + off
        // equivalent to our patch being moved r |-> r - off
        // keep in mind that we compute patch that we have to send
        // so we apply this offset on the patch we test against rather than ours
        vec off{
            per_vec.x() * periodicity_vec.x(),
            per_vec.y() * periodicity_vec.y(),
            per_vec.z() * periodicity_vec.z(),
        };

        bool has_off = !(
            (periodicity_vec.x() == 0) && (periodicity_vec.y() == 0) && (periodicity_vec.z() == 0));

        sycl::buffer<CommInd<flt>, 2> cbuf = generator::compute_buf_interact(
            sched, gen, sptree, -off, has_off, interact_crit, args...);

        {
            auto interface_list = sycl::host_accessor{cbuf, sycl::read_only};

            for (u64 i = 0; i < local_pcount; i++) {
                for (u64 j = 0; j < global_pcount; j++) {

                    if (interface_list[{i, j}].sender_patch_id == u64_max) {
                        break;
                    }
                    CommInd tmp = interface_list[{i, j}];

                    CommListingSend tmp_push;
                    tmp_push.applied_offset        = off;
                    tmp_push.periodicity_vector    = periodicity_vec;
                    tmp_push.local_patch_idx_send  = tmp.local_patch_idx_send;
                    tmp_push.global_patch_idx_recv = tmp.global_patch_idx_recv;
                    tmp_push.sender_patch_id       = tmp.sender_patch_id;
                    tmp_push.receiver_patch_id     = tmp.receiver_patch_id;

                    tmp_push.receiver_box_min = tmp.receiver_box_min;
                    tmp_push.receiver_box_max = tmp.receiver_box_max;

                    interf_send_map.push_back(std::move(tmp_push));

                    shamlog_debug_sycl_ln(
                        "Interfaces", "found : ", tmp.sender_patch_id, "->", tmp.receiver_patch_id);
                }
            }
        }
    };

    interf_send_map.clear();

    // TODO rethink this part to be able to use fixed bc for grid
    // probably one implementation of the whole thing for each boundary condition and then move user
    // through

    if (bc.has_outdomain_object()) {
        if (bc.periodic_search_min_vec.has_value() && bc.periodic_search_max_vec.has_value()) {

            u32_3 min = bc.periodic_search_min_vec.value();
            u32_3 max = bc.periodic_search_max_vec.value();

            for (u32 x = min.x(); x < max.x(); x++) {
                for (u32 y = min.y(); y < max.y(); y++) {
                    for (u32 z = min.z(); z < max.z(); z++) {
                        append_interface({x, y, z});
                    }
                }
            }
        } else {
            throw "Periodic search range not set";
        }
    } else {
        append_interface({0, 0, 0});
    }

    shamlog_debug_ln("Interfacehandler", "found", interf_send_map.size(), "interfaces");

    // then cutted make trees

    // Before impl this we have to code fullTreeFields (the local version (aka without interfaces))
    // //Tree cutter class allow more granularity over data

    for (CommListingSend &comm : interf_send_map) {

        auto &rtree = rtrees[comm.sender_patch_id];

        u32 total_count = rtree->tree_struct.internal_cell_count
                          + rtree->tree_reduced_morton_codes.tree_leaf_count;
        sycl::range<1> range_tree{total_count};

        shamlog_debug_sycl_ln("Radixtree", "computing valid node buf");

        auto init_valid_buf_val_unrolled = [&](auto... copied_vals) -> sycl::buffer<u8> {
            sycl::buffer<u8> valid_node = sycl::buffer<u8>(total_count);

            sycl::range<1> range_tree{total_count};

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_valid_node{valid_node, cgh, sycl::write_only, sycl::no_init};

                sycl::accessor acc_pos_cell_min{
                    *rtree->tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
                sycl::accessor acc_pos_cell_max{
                    *rtree->tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};

                InteractCrit cd = interact_crit;

                vec test_lbox_min = comm.receiver_box_min;
                vec test_lbox_max = comm.receiver_box_max;

                u32 cur_patch_idx = comm.local_patch_idx_send;

                u32 test_patch_idx = comm.global_patch_idx_recv;

                bool is_off_not_bull
                    = !((comm.applied_offset.x() == 0) && (comm.applied_offset.y() == 0)
                        && (comm.applied_offset.z() == 0));

                auto with_accessor = [=, &cgh](auto... accs_tree) {
                    cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                        auto cur_lbox_min = acc_pos_cell_min[item];
                        auto cur_lbox_max = acc_pos_cell_max[item];

                        acc_valid_node[item] = interact_crit::utils::interact_cd_cell_patch_domain(
                            cd,
                            is_off_not_bull,
                            cur_lbox_min,
                            cur_lbox_max,
                            test_lbox_min,
                            test_lbox_max,
                            accs_tree[item]...,
                            copied_vals...);
                    });
                };

                with_accessor(sycl::accessor{
                    *args.patch_tree_fields[comm.sender_patch_id]->radix_tree_field_buf,
                    cgh,
                    sycl::read_only}...);
            });

            return std::move(valid_node);
        };

        auto get_val = [&](auto field) {
            sycl::host_accessor acc{field.patch_field.buf_global};

            return acc[comm.global_patch_idx_recv];
        };

        auto buf = init_valid_buf_val_unrolled(get_val(args)...);

        shamlog_debug_ln(
            "InterfaceHandler",
            "gen tree for interf :",
            comm.sender_patch_id,
            "->",
            comm.receiver_patch_id);
        CutTree out(rtree->cut_tree(shamsys::instance::get_compute_queue(), buf));

        tree_send_map.list_rtree.push_back(
            std::make_unique<RadixTree<u_morton, vec>>(std::move(out.rtree)));
        tree_send_map.list_pdat_extract_id.push_back(std::move(out.pdat_extract_id));
        tree_send_map.list_new_node_id_to_old.push_back(std::move(out.new_node_id_to_old));
    }
}

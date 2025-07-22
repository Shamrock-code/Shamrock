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
 * @file SerialPatchTree.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

//%Impl status : Should rewrite

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/legacy/patch/utility/patch_field.hpp"
#include "shamrock/legacy/patch/utility/patch_reduc_tree.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include <array>
#include <tuple>
#include <vector>

template<class fp_prec_vec>
class SerialPatchTree {
    public:
    using PtNode = shamrock::scheduler::SerialPatchNode<fp_prec_vec>;

    using PatchTree = shamrock::scheduler::PatchTree;

    // TODO use unique pointer instead
    u32 root_count = 0;
    std::unique_ptr<sycl::buffer<PtNode>> serial_tree_buf;
    std::unique_ptr<sycl::buffer<u64>> linked_patch_ids_buf;

    inline void attach_buf() {
        if (bool(serial_tree_buf))
            throw shambase::make_except_with_loc<std::runtime_error>(
                "serial_tree_buf is already allocated");
        if (bool(linked_patch_ids_buf))
            throw shambase::make_except_with_loc<std::runtime_error>(
                "linked_patch_ids_buf is already allocated");

        serial_tree_buf
            = std::make_unique<sycl::buffer<PtNode>>(serial_tree.data(), serial_tree.size());
        linked_patch_ids_buf
            = std::make_unique<sycl::buffer<u64>>(linked_patch_ids.data(), linked_patch_ids.size());
    }

    inline void detach_buf() {
        if (!bool(serial_tree_buf))
            throw shambase::make_except_with_loc<std::runtime_error>(
                "serial_tree_buf wasn't allocated");
        if (!bool(linked_patch_ids_buf))
            throw shambase::make_except_with_loc<std::runtime_error>(
                "linked_patch_ids_buf wasn't allocated");

        serial_tree_buf.reset();
        linked_patch_ids_buf.reset();
    }

    private:
    u32 level_count = 0;

    std::vector<PtNode> serial_tree;
    std::vector<u64> linked_patch_ids;
    std::vector<u64> roots_ids;

    void build_from_patch_tree(
        PatchTree &ptree, const shamrock::patch::PatchCoordTransform<fp_prec_vec> box_transform);

    public:
    inline void print_status() {
        if (shamcomm::world_rank() == 0) {
            for (PtNode n : serial_tree) {
                logger::raw_ln(
                    n.box_min,
                    n.box_max,
                    "[",
                    n.childs_id[0],
                    n.childs_id[1],
                    n.childs_id[2],
                    n.childs_id[3],
                    n.childs_id[4],
                    n.childs_id[5],
                    n.childs_id[6],
                    n.childs_id[7],
                    "]");
            }
        }
    }

    inline SerialPatchTree(
        PatchTree &ptree, const shamrock::patch::PatchCoordTransform<fp_prec_vec> box_transform) {
        StackEntry stack_loc{};
        build_from_patch_tree(ptree, box_transform);
    }

    inline void host_for_each_leafs(
        std::function<bool(u64, PtNode pnode)> interact_cd,
        std::function<void(u64, PtNode)> found_case) {
        StackEntry stack_loc{false};

        sycl::host_accessor tree{shambase::get_check_ref(serial_tree_buf), sycl::read_only};
        sycl::host_accessor lpid{shambase::get_check_ref(linked_patch_ids_buf), sycl::read_only};

        std::stack<u64> id_stack;

        for (u64 root : roots_ids) {
            id_stack.push(root);
        }

        while (!id_stack.empty()) {
            u64 cur_id = id_stack.top();
            id_stack.pop();
            PtNode cur_p = tree[cur_id];

            bool interact = interact_cd(cur_id, cur_p);

            if (interact) {
                u64 linked_id = lpid[cur_id];
                if (linked_id != u64_max) {
                    found_case(linked_id, cur_p);
                } else {
                    id_stack.push(cur_p.childs_id[0]);
                    id_stack.push(cur_p.childs_id[1]);
                    id_stack.push(cur_p.childs_id[2]);
                    id_stack.push(cur_p.childs_id[3]);
                    id_stack.push(cur_p.childs_id[4]);
                    id_stack.push(cur_p.childs_id[5]);
                    id_stack.push(cur_p.childs_id[6]);
                    id_stack.push(cur_p.childs_id[7]);
                }
            }
        }
    }

    /**
     * @brief accesor to the number of level in the tree
     *
     * @return const u32& number of level
     */
    inline const u32 &get_level_count() { return level_count; }

    /**
     * @brief accesor to the number of element in the tree
     *
     * @return const u32& number of element
     */
    inline u32 get_element_count() { return serial_tree.size(); }

    inline static SerialPatchTree<fp_prec_vec> build(PatchScheduler &sched) {
        return SerialPatchTree<fp_prec_vec>(
            sched.patch_tree, sched.get_patch_transform<fp_prec_vec>());
    }

    template<class type, class reduc_func>
    inline PatchFieldReduction<type>
    reduce_field(sycl::queue &queue, PatchScheduler &sched, legacy::PatchField<type> &pfield) {

        PatchFieldReduction<type> predfield;

        std::cout << "resize to " << get_element_count() << std::endl;
        predfield.tree_field.resize(get_element_count());

        {
            sycl::host_accessor lpid{*linked_patch_ids_buf, sycl::read_only};

            // init reduction
            std::unordered_map<u64, u64> &idp_to_gid = sched.patch_list.id_patch_to_global_idx;
            for (u64 idx = 0; idx < get_element_count(); idx++) {
                predfield.tree_field[idx]
                    = (lpid[idx] != u64_max) ? pfield.global_values[idp_to_gid[lpid[idx]]] : type();

                // std::cout << " el " << idx << " " << predfield.tree_field[idx]  << std::endl;
            }
        }

        // std::cout << "predfield.attach_buf();" << std::endl;

        predfield.attach_buf();

        sycl::range<1> range{get_element_count()};

        u32 end_loop = get_level_count();

        for (u32 level = 0; level < end_loop; level++) {

            // {
            //     auto f = predfield.tree_field_buf->template
            //     get_access<sycl::access::mode::read>(); std::cout << "["; for (u64 idx = 0; idx <
            //     get_element_count() ; idx ++) {
            //         std::cout  << f[idx] << ",";
            //     }
            //     std::cout << std::endl;
            // }

            std::cout << "queue submit : " << level << " " << end_loop << " " << (level < end_loop)
                      << std::endl;
            queue.submit([&](sycl::handler &cgh) {
                auto tree
                    = this->serial_tree_buf->template get_access<sycl::access::mode::read>(cgh);

                auto f
                    = predfield.tree_field_buf->template get_access<sycl::access::mode::read_write>(
                        cgh);

                cgh.parallel_for<class OctreeReduction>(range, [=](sycl::item<1> item) {
                    u64 i = (u64) item.get_id(0);

                    u64 idx_c0 = tree[i].childs_id0;
                    u64 idx_c1 = tree[i].childs_id1;
                    u64 idx_c2 = tree[i].childs_id2;
                    u64 idx_c3 = tree[i].childs_id3;
                    u64 idx_c4 = tree[i].childs_id4;
                    u64 idx_c5 = tree[i].childs_id5;
                    u64 idx_c6 = tree[i].childs_id6;
                    u64 idx_c7 = tree[i].childs_id7;

                    if (idx_c0 != u64_max) {
                        f[i] = reduc_func::reduce(
                            f[idx_c0],
                            f[idx_c1],
                            f[idx_c2],
                            f[idx_c3],
                            f[idx_c4],
                            f[idx_c5],
                            f[idx_c6],
                            f[idx_c7]);
                    }
                });
            });
        }
        // {
        //     auto f = predfield.tree_field_buf->template get_access<sycl::access::mode::read>();
        //     std::cout << "[";
        //     for (u64 idx = 0; idx < get_element_count() ; idx ++) {
        //         std::cout  << f[idx] << ",";
        //     }
        //     std::cout << std::endl;
        // }

        return predfield;
    }

    template<class T, class Func>
    inline shamrock::patch::PatchtreeField<T> make_patch_tree_field(
        PatchScheduler &sched,
        sycl::queue &queue,
        shamrock::patch::PatchField<T> pfield,
        Func &&reducer) {
        shamrock::patch::PatchtreeField<T> ptfield;
        ptfield.allocate(get_element_count());

        {
            sycl::host_accessor lpid{
                shambase::get_check_ref(linked_patch_ids_buf), sycl::read_only};
            sycl::host_accessor tree_field{
                shambase::get_check_ref(ptfield.internal_buf), sycl::write_only, sycl::no_init};

            // init reduction
            std::unordered_map<u64, u64> &idp_to_gid = sched.patch_list.id_patch_to_global_idx;
            for (u64 idx = 0; idx < get_element_count(); idx++) {
                tree_field[idx] = (lpid[idx] != u64_max) ? pfield.get(lpid[idx]) : T();
            }
        }

        sycl::range<1> range{get_element_count()};
        u32 end_loop = get_level_count();

        for (u32 level = 0; level < end_loop; level++) {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor tree{shambase::get_check_ref(serial_tree_buf), cgh, sycl::read_only};
                sycl::accessor f{
                    shambase::get_check_ref(ptfield.internal_buf), cgh, sycl::read_write};

                cgh.parallel_for(range, [=](sycl::item<1> item) {
                    u64 i = (u64) item.get_id(0);

                    std::array<u64, 8> n = tree[i].childs_id;

                    if (n[0] != u64_max) {
                        f[i] = reducer(
                            f[n[0]], f[n[1]], f[n[2]], f[n[3]], f[n[4]], f[n[5]], f[n[6]], f[n[7]]);
                    }
                });
            });
        }
        return ptfield;
    }

    inline void dump_dat() {
        for (u64 idx = 0; idx < get_element_count(); idx++) {
            std::cout << idx << " (" << serial_tree[idx].childs_id[0] << ", "
                      << serial_tree[idx].childs_id[1] << ", " << serial_tree[idx].childs_id[2]
                      << ", " << serial_tree[idx].childs_id[3] << ", "
                      << serial_tree[idx].childs_id[4] << ", " << serial_tree[idx].childs_id[5]
                      << ", " << serial_tree[idx].childs_id[6] << ", "
                      << serial_tree[idx].childs_id[7] << ")";

            std::cout << " (" << serial_tree[idx].box_min.x() << ", "
                      << serial_tree[idx].box_min.y() << ", " << serial_tree[idx].box_min.z()
                      << ")";

            std::cout << " (" << serial_tree[idx].box_max.x() << ", "
                      << serial_tree[idx].box_max.y() << ", " << serial_tree[idx].box_max.z()
                      << ")";

            std::cout << " = " << linked_patch_ids[idx];

            std::cout << std::endl;
        }
    }

    sycl::buffer<u64> compute_patch_owner(
        sham::DeviceScheduler_ptr dev_sched,
        sham::DeviceBuffer<fp_prec_vec> &position_buffer,
        u32 len);
};

template<class vec>
sycl::buffer<u64> SerialPatchTree<vec>::compute_patch_owner(
    sham::DeviceScheduler_ptr dev_sched, sham::DeviceBuffer<vec> &position_buffer, u32 len) {
    sycl::buffer<u64> new_owned_id(len);

    using namespace shamrock::patch;

    sycl::buffer<u64> roots = shamalgs::vec_to_buf(roots_ids);

    auto &q = dev_sched->get_queue();

    sham::EventList depends_list;
    auto pos = position_buffer.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        sycl::accessor tnode{shambase::get_check_ref(serial_tree_buf), cgh, sycl::read_only};
        sycl::accessor linked_node_id{
            shambase::get_check_ref(linked_patch_ids_buf), cgh, sycl::read_only};
        sycl::accessor roots_id{roots, cgh, sycl::read_only};
        sycl::accessor new_id{new_owned_id, cgh, sycl::write_only, sycl::no_init};

        u32 root_cnt = roots_id.size();
        auto max_lev = get_level_count();

        using PtNode = shamrock::scheduler::SerialPatchNode<vec>;

        cgh.parallel_for(sycl::range(len), [=](sycl::item<1> item) {
            u32 i = (u32) item.get_id(0);

            auto xyz = pos[i];

            u64 current_node = 0;

            // find the correct root to start the search
            for (u32 iroot = 0; iroot < root_cnt; iroot++) {
                u32 root_id      = roots_id[iroot];
                PtNode root_node = tnode[root_id];

                if (Patch::is_in_patch_converted(xyz, root_node.box_min, root_node.box_max)) {
                    current_node = root_id;
                    break;
                }
            }

            u64 result_node = u64_max;

            for (u32 step = 0; step < max_lev + 1; step++) {
                PtNode cur_node = tnode[current_node];

                if (cur_node.childs_id[0] != u64_max) {

                    if (Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[0]].box_min,
                            tnode[cur_node.childs_id[0]].box_max)) {
                        current_node = cur_node.childs_id[0];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[1]].box_min,
                                   tnode[cur_node.childs_id[1]].box_max)) {
                        current_node = cur_node.childs_id[1];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[2]].box_min,
                                   tnode[cur_node.childs_id[2]].box_max)) {
                        current_node = cur_node.childs_id[2];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[3]].box_min,
                                   tnode[cur_node.childs_id[3]].box_max)) {
                        current_node = cur_node.childs_id[3];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[4]].box_min,
                                   tnode[cur_node.childs_id[4]].box_max)) {
                        current_node = cur_node.childs_id[4];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[5]].box_min,
                                   tnode[cur_node.childs_id[5]].box_max)) {
                        current_node = cur_node.childs_id[5];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[6]].box_min,
                                   tnode[cur_node.childs_id[6]].box_max)) {
                        current_node = cur_node.childs_id[6];
                    } else if (Patch::is_in_patch_converted(
                                   xyz,
                                   tnode[cur_node.childs_id[7]].box_min,
                                   tnode[cur_node.childs_id[7]].box_max)) {
                        current_node = cur_node.childs_id[7];
                    }

                } else {

                    result_node = linked_node_id[current_node];
                    break;
                }
            }

            if constexpr (false) {
                PtNode cur_node = tnode[current_node];
                if (xyz[0] == 0 && xyz[1] == 0 && xyz[2] == 0) {
                    logger::raw(shambase::format(
                        "{:5} ({}) -> {} [{} {}]\n",
                        i,
                        Patch::is_in_patch_converted(xyz, cur_node.box_min, cur_node.box_max),
                        xyz,
                        cur_node.box_min,
                        cur_node.box_max));
                }
            }

            new_id[i] = result_node;
        });
    });

    position_buffer.complete_event_state(e);

    return new_owned_id;
}

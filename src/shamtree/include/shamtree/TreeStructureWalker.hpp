// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TreeStructureWalker.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "TreeStructure.hpp"

namespace shamrock::tree {

    enum WalkPolicy { Recompute, Cache };

    namespace details {
        template<WalkPolicy policy, class u_morton, class InteractCrit>
        class TreeStructureWalkerPolicy;
    } // namespace details

    template<WalkPolicy policy, class u_morton, class InteractCrit>
    class TreeStructureWalker {
        public:
        details::TreeStructureWalkerPolicy<policy, u_morton, InteractCrit> walker;

        using AccessedWalker =
            typename details::TreeStructureWalkerPolicy<policy, u_morton, InteractCrit>::Accessed;

        TreeStructureWalker(TreeStructure<u_morton> &str, u32 walker_count, InteractCrit &&crit)
            : walker(str, walker_count, std::forward<InteractCrit>(crit)) {}

        inline void generate() { walker.generate(); }

        inline AccessedWalker get_access(sycl::handler &device_handle) {
            return walker.get_access(device_handle);
        }
    };

    template<WalkPolicy policy, class u_morton, class InteractCrit>
    static TreeStructureWalker<policy, u_morton, InteractCrit> generate_walk(
        TreeStructure<u_morton> &str, u32 walker_count, InteractCrit &&crit) {
        TreeStructureWalker<policy, u_morton, InteractCrit> walk(
            str, walker_count, std::forward<InteractCrit>(crit));
        walk.generate();
        return walk;
    }

} // namespace shamrock::tree

namespace shamrock::tree::details {

    template<class u_morton, class InteractCrit>
    class TreeStructureWalkerPolicy<Recompute, u_morton, InteractCrit> {
        public:
        TreeStructure<u_morton> &tree_struct;
        InteractCrit crit;
        u32 walker_count;

        class Accessed {

            public:
            using IntCritAcc  = typename InteractCrit::Access;
            using IntCritVals = typename IntCritAcc::ObjectValues;

            private:
            sycl::range<1> walkers_range;

            u32 leaf_offset;

            sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
            sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
            sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
            sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;

            IntCritAcc criterion_acc;

            static constexpr auto get_tree_depth = []() -> u32 {
                if constexpr (std::is_same<u_morton, u32>::value) {
                    return 32;
                }
                if constexpr (std::is_same<u_morton, u64>::value) {
                    return 64;
                }
                return 0;
            };

            static constexpr u32 tree_depth = get_tree_depth();
            static constexpr u32 _nindex    = 4294967295;

            bool one_cell_mode;

            public:
            Accessed(
                TreeStructure<u_morton> &tree_struct,
                u32 walker_count,
                sycl::handler &device_handle,
                InteractCrit crit)
                : rchild_id{*tree_struct.buf_rchild_id, device_handle, sycl::read_only},
                  lchild_id{*tree_struct.buf_lchild_id, device_handle, sycl::read_only},
                  rchild_flag{*tree_struct.buf_rchild_flag, device_handle, sycl::read_only},
                  lchild_flag{*tree_struct.buf_lchild_flag, device_handle, sycl::read_only},
                  criterion_acc(crit, device_handle), walkers_range{walker_count},
                  leaf_offset(tree_struct.internal_cell_count),
                  one_cell_mode(tree_struct.one_cell_mode) {}

            inline sycl::range<1> get_sycl_range() { return walkers_range; }

            inline IntCritAcc criterion() const { return criterion_acc; }

            template<class FuncNodeFound, class FuncNodeReject>
            inline void for_each_node(
                sycl::item<1> id,
                IntCritVals int_values,
                FuncNodeFound &&found_case,
                FuncNodeReject &&reject_case) const;
        };

        inline void generate() {}

        TreeStructureWalkerPolicy(
            TreeStructure<u_morton> &str, u32 walker_count, InteractCrit &&crit)
            : tree_struct(str), walker_count(walker_count), crit(crit) {}

        inline Accessed get_access(sycl::handler &device_handle) {
            return Accessed(tree_struct, walker_count, device_handle, crit);
        }
    };
} // namespace shamrock::tree::details

template<class u_morton, class InteractCrit>
template<class FuncNodeFound, class FuncNodeReject>
inline void shamrock::tree::details::
    TreeStructureWalkerPolicy<shamrock::tree::Recompute, u_morton, InteractCrit>::Accessed::
        for_each_node(
            sycl::item<1> id,
            IntCritVals int_values,
            FuncNodeFound &&found_case,
            FuncNodeReject &&reject_case) const {

    u32 stack_cursor = tree_depth - 1;
    std::array<u32, tree_depth> id_stack;

    // Should be unrequired considering the change made to the tree building
    // if (one_cell_mode) {
    //
    //     bool valid_root = InteractCrit::criterion(0, criterion_tree_f_acc, int_values);
    //
    //     if (valid_root) {
    //         found_case(leaf_offset, 0);
    //     } else {
    //         reject_case(0);
    //     }
    //
    //     return;
    // }

    id_stack[stack_cursor] = 0;

    while (stack_cursor < tree_depth) {

        u32 current_node_id    = id_stack[stack_cursor];
        id_stack[stack_cursor] = _nindex;
        stack_cursor++;

        bool cur_id_valid = InteractCrit::criterion(current_node_id, criterion_acc, int_values);

        if (cur_id_valid) {

            // leaf and cell can interact
            if (current_node_id >= leaf_offset) {

                found_case(current_node_id, current_node_id - leaf_offset);

                // can interact not leaf => stack
            } else {

                u32 lid = lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                u32 rid = rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

                id_stack[stack_cursor - 1] = rid;
                stack_cursor--;

                id_stack[stack_cursor - 1] = lid;
                stack_cursor--;
            }
        } else {
            // grav
            reject_case(current_node_id);
        }
    }
}

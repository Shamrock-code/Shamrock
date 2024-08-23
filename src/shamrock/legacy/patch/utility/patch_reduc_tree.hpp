// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
/**
 * @file patch_reduc_tree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include <stdexcept>
#include <vector>

template<class type>
class PatchFieldReduction {
    public:
    std::vector<type> tree_field;
    sycl::buffer<type> *tree_field_buf = nullptr;

    inline void attach_buf() {
        if (tree_field_buf != nullptr)
            throw shambase::make_except_with_loc<std::runtime_error>(
                "tree_field_buf is already allocated");
        tree_field_buf = new sycl::buffer<type>(tree_field.data(), tree_field.size());
    }

    inline void detach_buf() {
        if (tree_field_buf == nullptr)
            throw shambase::make_except_with_loc<std::runtime_error>(
                "tree_field_buf wasn't allocated");
        delete tree_field_buf;
        tree_field_buf = nullptr;
    }

    // inline void octtree_reduction(
    //     sycl::queue & queue,
    //     SerialPatchTree<box_vectype> & sptree,
    //     SchedulerMPI & sched){

    //     std::unordered_map<u64,u64> & idp_to_gid = sched.patch_list.id_patch_to_global_idx;

    //     sycl::range<1> range{sptree.get_element_count()};

    //     for (u32 level = 0; level < sptree.get_level_count(); level ++) {
    //         queue.submit([&](sycl::handler &cgh) {

    //             cgh.parallel_for<class OctreeReduction>(range, [=](sycl::item<1> item) {
    //                 u64 i = (u64)item.get_id(0);

    //             });
    //         });
    //     }

    // }
};

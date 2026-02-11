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
 * @file compute_tree_mass_moments.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shammath/AABB.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, class Umorton, u32 moment_order>
    inline shamtree::KarrasRadixTreeFieldMultiVar<shambase::VecComponent<Tvec>>
    compute_tree_mass_moments(
        shamtree::CompressedLeafBVH<Umorton, Tvec, 3> &bvh,
        const sham::DeviceBuffer<Tvec> &xyz,
        shambase::VecComponent<Tvec> gpart_mass) {

        __shamrock_stack_entry();

        using Tscal       = shambase::VecComponent<Tvec>;
        using MassMoments = shammath::SymTensorCollection<Tscal, 0, moment_order>;
        static constexpr u32 mass_moment_terms = MassMoments::num_component;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        // compute moments in leaves
        auto mass_moments_tree = shamtree::prepare_karras_radix_tree_field_multi_var<Tscal>(
            bvh.structure,
            shamtree::new_empty_karras_radix_tree_field_multi_var<Tscal>(mass_moment_terms));

        // fill the leaves with the mass moments
        auto cell_it = bvh.reduced_morton_set.get_leaf_cell_iterator();
        auto fill_leafs
            = [&](shamtree::KarrasRadixTreeFieldMultiVar<Tscal> &tree_field, u32 leaf_offset) {
                  sham::kernel_call(
                      q,
                      sham::MultiRef{xyz, cell_it, bvh.aabbs.buf_aabb_min, bvh.aabbs.buf_aabb_max},
                      sham::MultiRef{tree_field.buf_field},
                      bvh.structure.get_leaf_count(),
                      [leaf_offset, gpart_mass](
                          u32 i,
                          const Tvec *xyz,
                          auto cell_iter,
                          const Tvec *aabb_min,
                          const Tvec *aabb_max,
                          Tscal *mass_moments_scal) {
                          // Init with the min value of the type
                          MassMoments Q_n_B = MassMoments::zeros();

                          u32 cell_id = i + leaf_offset;
                          shammath::AABB<Tvec> cell_aabb
                              = shammath::AABB<Tvec>{aabb_min[cell_id], aabb_max[cell_id]};

                          Tvec s_B = cell_aabb.get_center();

                          cell_iter.for_each_in_leaf_cell(i, [&](u32 j) {
                              Q_n_B += MassMoments::from_vec(xyz[j] - s_B);
                          });

                          Q_n_B *= gpart_mass;

                          Tscal *ptr_store
                              = mass_moments_scal + (i + leaf_offset) * MassMoments::num_component;
                          Q_n_B.store(ptr_store, 0);
                      });
              };

        fill_leafs(mass_moments_tree, bvh.structure.get_internal_cell_count());

        // propagate the moments upward
        u32 int_cell_count = bvh.structure.get_internal_cell_count();

        if (int_cell_count == 0) {
            return mass_moments_tree;
        }

        auto step = [&]() {
            auto traverser = bvh.structure.get_structure_traverser();

            sham::kernel_call(
                q,
                sham::MultiRef{traverser, bvh.aabbs.buf_aabb_min, bvh.aabbs.buf_aabb_max},
                sham::MultiRef{mass_moments_tree.buf_field},
                int_cell_count,
                [=](u32 gid,
                    auto tree_traverser,
                    const Tvec *aabb_min,
                    const Tvec *aabb_max,
                    Tscal *__restrict moments) {
                    u32 left_child  = tree_traverser.get_left_child(gid);
                    u32 right_child = tree_traverser.get_right_child(gid);

                    Tscal *ptr_left  = moments + left_child * MassMoments::num_component;
                    Tscal *ptr_right = moments + right_child * MassMoments::num_component;

                    Tvec left_center
                        = shammath::AABB<Tvec>{aabb_min[left_child], aabb_max[left_child]}
                              .get_center();
                    Tvec right_center
                        = shammath::AABB<Tvec>{aabb_min[right_child], aabb_max[right_child]}
                              .get_center();
                    Tvec new_center
                        = shammath::AABB<Tvec>{aabb_min[gid], aabb_max[gid]}.get_center();

                    MassMoments Q_n_B_left  = MassMoments::load(ptr_left, 0);
                    MassMoments Q_n_B_right = MassMoments::load(ptr_right, 0);

                    // offset the moments and add them
                    MassMoments Q_n_B_combined = MassMoments::zeros();
                    Q_n_B_combined
                        += shamphys::offset_multipole(Q_n_B_left, left_center, new_center);
                    Q_n_B_combined
                        += shamphys::offset_multipole(Q_n_B_right, right_center, new_center);

                    Tscal *ptr_store = moments + gid * MassMoments::num_component;
                    Q_n_B_combined.store(ptr_store, 0);
                });
        };

        for (u32 i = 0; i < bvh.structure.tree_depth; i++) {
            step();
        }

        return mass_moments_tree;
    }
} // namespace shammodels::sph::modules

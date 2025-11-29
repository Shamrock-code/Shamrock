// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SGFMMPlummer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/self_gravity/SGFMMPlummer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shammodels/sph/modules/self_gravity/compute_tree_mass_moments.hpp"
#include "shamphys/fmm/GreenFuncGravCartesian.hpp"
#include "shamphys/fmm/contract_grav_moment.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, u32 mm_order>
    void SGFMMPlummer<Tvec, mm_order>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        using Umorton = u32;
        using RTree   = shamtree::CompressedLeafBVH<Umorton, Tvec, 3>;

        auto edges = get_edges();

        edges.field_axyz_ext.ensure_sizes(edges.sizes.indexes);

        if (edges.sizes.indexes.get_ids().size() != 1) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Self gravity direct mode only supports one patch so far, current number "
                "of patches is : "
                + std::to_string(edges.sizes.indexes.get_ids().size()));
        }

        Tscal G          = edges.constant_G.data;
        Tscal gpart_mass = edges.gpart_mass.data;

        Tscal gravitational_softening = epsilon;

        using MassMoments = shammath::SymTensorCollection<Tscal, 0, mm_order - 1>;
        static constexpr u32 mass_moment_terms = MassMoments::num_component;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes.indexes
            .for_each(
                [&](u64 id, const u64 &n) {
                    PatchDataField<Tvec> &xyz      = edges.field_xyz.get_field(id);
                    PatchDataField<Tvec> &axyz_ext = edges.field_axyz_ext.get_field(id);

                    Tvec bmax = xyz.compute_max();
                    Tvec bmin = xyz.compute_min();
                    shammath::AABB<Tvec> aabb(bmin, bmax);

                    // build the tree
                    auto bvh = RTree::make_empty(dev_sched);
                    bvh.rebuild_from_positions(
                        xyz.get_buf(), xyz.get_obj_cnt(), aabb, reduction_level);

                    // compute moments in leaves
                    auto mass_moments_tree = compute_tree_mass_moments<Tvec, Umorton, mm_order - 1>(
                        bvh, xyz.get_buf(), gpart_mass);

                    // compute the FMM force in leaves on particles
                    auto obj_it = bvh.get_object_iterator();
                    sham::kernel_call(
                        q,
                        sham::MultiRef{xyz.get_buf(), obj_it, mass_moments_tree.buf_field},
                        sham::MultiRef{axyz_ext.get_buf()},
                        bvh.structure.get_leaf_count(),
                        [theta_crit = theta_crit, gravitational_softening, gpart_mass, G](
                            u32 ileaf,
                            const Tvec *xyz,
                            auto particle_looper,
                            const Tscal *mass_moments_scal,
                            Tvec *axyz_ext) {
                            auto &tree_traverser = particle_looper.tree_traverser;
                            auto &cell_iterator  = particle_looper.cell_iterator;

                            u32 leaf_id_tree = ileaf + tree_traverser.tree_traverser.offset_leaf;
                            shammath::AABB<Tvec> box_A_AABB = shammath::AABB<Tvec>{
                                tree_traverser.aabb_min[leaf_id_tree],
                                tree_traverser.aabb_max[leaf_id_tree]};

                            Tvec s_A     = box_A_AABB.get_center();
                            Tvec delta_A = box_A_AABB.delt();
                            Tscal sz_A   = sham::max_component(delta_A) / 2;

                            auto dM_k = shammath::SymTensorCollection<Tscal, 1, mm_order>::zeros();

                            tree_traverser
                                .traverse_tree_base(
                                    [&](u32 node_id) -> bool {
                                        // mac

                                        shammath::AABB<Tvec> node_aabb = shammath::AABB<Tvec>{
                                            tree_traverser.aabb_min[node_id],
                                            tree_traverser.aabb_max[node_id]};

                                        Tvec s_B     = node_aabb.get_center();
                                        Tvec delta_B = node_aabb.delt();
                                        Tscal sz_B   = sham::max_component(delta_B) / 2;

                                        Tvec r     = s_B - s_A;
                                        Tscal r_sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                                        Tscal theta = (r_sq == 0)
                                                          ? theta_crit * 2
                                                          : (sz_B + sz_A) * sycl::rsqrt(r_sq);

                                        return theta > theta_crit;
                                    },
                                    [&](u32 node_id) { // p2p case
                                        u32 leaf_id
                                            = node_id - tree_traverser.tree_traverser.offset_leaf;

                                        cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 j) {
                                            cell_iterator.for_each_in_leaf_cell(ileaf, [&](u32 i) {
                                                Tvec R            = xyz[j] - xyz[i];
                                                const Tscal r_inv = sycl::rsqrt(
                                                    R.x() * R.x() + R.y() * R.y() + R.z() * R.z()
                                                    + gravitational_softening);
                                                axyz_ext[i]
                                                    += G * gpart_mass * r_inv * r_inv * r_inv * R;
                                            });
                                        });

                                    },
                                    [&](u32 node_id) {
                                        // multipole case
                                        Tvec s_B = shammath::AABB<Tvec>{tree_traverser.aabb_min[node_id], tree_traverser.aabb_max[node_id]}.get_center();

                                        Tvec r_fmm = s_B - s_A;

                                        MassMoments Q_n_B = MassMoments::load(
                                            mass_moments_scal
                                                + node_id * MassMoments::num_component,
                                            0);

                                        auto D_n
                                            = shamphys::GreenFuncGravCartesian<Tscal, 1, mm_order>::
                                                get_der_tensors(r_fmm);

                                        dM_k += shamphys::get_dM_mat(D_n, Q_n_B);
                                    });

                            // at the end apply the grav moments on the particles
                            cell_iterator.for_each_in_leaf_cell(ileaf, [&](u32 i) {
                                Tvec a_i = xyz[i] - s_A;

                                auto a_k = shammath::SymTensorCollection<Tscal, 0, mm_order - 1>::
                                    from_vec(a_i);

                                axyz_ext[i]
                                    += -G
                                       * shamphys::contract_grav_moment_to_force<Tscal, mm_order>(
                                           a_k, dM_k);
                            });
                        });
                });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SGFMMPlummer<f64_3, 1>;
template class shammodels::sph::modules::SGFMMPlummer<f64_3, 2>;
template class shammodels::sph::modules::SGFMMPlummer<f64_3, 3>;
template class shammodels::sph::modules::SGFMMPlummer<f64_3, 4>;
template class shammodels::sph::modules::SGFMMPlummer<f64_3, 5>;

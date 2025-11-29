// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SGMMPlummer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/self_gravity/SGMMPlummer.hpp"
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
    void SGMMPlummer<Tvec, mm_order>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        using Umorton = u32;
        using RTree   = shamtree::CompressedLeafBVH<Umorton, Tvec, 3>;

        auto edges = get_edges();

        edges.field_axyz_ext.ensure_sizes(edges.sizes.indexes);

        if (edges.sizes.indexes.get_ids().size() != 1) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Self gravity MM mode only supports one patch so far, current number "
                "of patches is : "
                + std::to_string(edges.sizes.indexes.get_ids().size()));
        }

        Tscal G          = edges.constant_G.data;
        Tscal gpart_mass = edges.gpart_mass.data;

        Tscal gravitational_softening = epsilon * epsilon;

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

                    // compute the force on the particles
                    auto obj_it = bvh.get_object_iterator();
                    sham::kernel_call(
                        q,
                        sham::MultiRef{xyz.get_buf(), obj_it, mass_moments_tree.buf_field},
                        sham::MultiRef{axyz_ext.get_buf()},
                        n,
                        [theta_crit = theta_crit, gravitational_softening, gpart_mass, G](
                            u32 i,
                            const Tvec *xyz,
                            auto particle_looper,
                            const Tscal *mass_moments_scal,
                            Tvec *axyz_ext) {
                            Tvec xyz_i = xyz[i];

                            Tvec f_i{0.0f};

                            auto &tree_traverser = particle_looper.tree_traverser;
                            auto &cell_iterator  = particle_looper.cell_iterator;

                            tree_traverser
                                .traverse_tree_base(
                                    [&](u32 node_id) -> bool {
                                        // mac

                                        shammath::AABB<Tvec> node_aabb = shammath::AABB<Tvec>{
                                            tree_traverser.aabb_min[node_id],
                                            tree_traverser.aabb_max[node_id]};

                                        Tvec s_B = node_aabb.get_center();

                                        Tvec delta_B = node_aabb.delt();

                                        Tscal sz_B = sham::max_component(delta_B) / 2;

                                        Tvec r     = s_B - xyz_i;
                                        Tscal r_sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                                        Tscal theta = (r_sq == 0) ? theta_crit * 2
                                                                  : sz_B * sycl::rsqrt(r_sq);

                                        return theta > theta_crit;
                                    },
                                    [&](u32 node_id) { // p2p case
                                        u32 leaf_id
                                            = node_id - tree_traverser.tree_traverser.offset_leaf;
                                        cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 j) {
                                            Tvec R            = xyz[j] - xyz_i;
                                            const Tscal r_inv = sycl::rsqrt(
                                                R.x() * R.x() + R.y() * R.y() + R.z() * R.z()
                                                + gravitational_softening);
                                            f_i += gpart_mass * r_inv * r_inv * r_inv * R;
                                        });
                                    },
                                    [&](u32 node_id) {
                                        // multipole case
                                        Tvec s_B = shammath::AABB<Tvec>{tree_traverser.aabb_min[node_id], tree_traverser.aabb_max[node_id]}.get_center();

                                        Tvec r_fmm = s_B - xyz_i;

                                        MassMoments Q_n_B = MassMoments::load(
                                            mass_moments_scal
                                                + node_id * MassMoments::num_component,
                                            0);
                                        auto D_n
                                            = shamphys::GreenFuncGravCartesian<Tscal, 1, mm_order>::
                                                get_der_tensors(r_fmm);

                                        f_i -= shamphys::
                                            contract_grav_moment_to_force<Tscal, mm_order>(
                                                Q_n_B, D_n);
                                    });

                            axyz_ext[i] += f_i * G;
                        });
                });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SGMMPlummer<f64_3, 1>;
template class shammodels::sph::modules::SGMMPlummer<f64_3, 2>;
template class shammodels::sph::modules::SGMMPlummer<f64_3, 3>;
template class shammodels::sph::modules::SGMMPlummer<f64_3, 4>;
template class shammodels::sph::modules::SGMMPlummer<f64_3, 5>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SGDirectPlummer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/self_gravity/SGDirectPlummer.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    void SGDirectPlummer<Tvec>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        if (edges.sizes.indexes.get_ids().size() != 1) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Self gravity direct mode only supports one patch so far, current number "
                "of patches is : "
                + std::to_string(edges.sizes.indexes.get_ids().size()));
        }

        const Tscal G          = edges.constant_G.data;
        const Tscal gpart_mass = edges.gpart_mass.data;

        const Tscal gravitational_softening = epsilon * epsilon;

        edges.sizes.indexes.for_each([&](u64 id, const u64 &n) {
            PatchDataField &xyz      = edges.field_xyz.get_field(id);
            PatchDataField &axyz_ext = edges.field_axyz_ext.get_field(id);

            if (reference_mode) {
                std::vector<Tvec> xyz_vec      = xyz.get_buf().copy_to_stdvec();
                std::vector<Tvec> axyz_ext_vec = axyz_ext.get_buf().copy_to_stdvec();

                for (u64 i = 0; i < n; i++) {
                    Tvec force{0.0f};
                    for (u64 j = 0; j < n; j++) {
                        if (i == j) {
                            continue;
                        }

                        Tvec R            = xyz_vec[j] - xyz_vec[i];
                        const Tscal r_inv = sycl::rsqrt(
                            R.x() * R.x() + R.y() * R.y() + R.z() * R.z()
                            + gravitational_softening);
                        force += gpart_mass * r_inv * r_inv * r_inv * R;
                    }
                    axyz_ext_vec[i] += force * G;
                }

                axyz_ext.get_buf().copy_from_stdvec(axyz_ext_vec);

            } else {

                const u32 group_size    = 32;
                const u32 group_cnt     = shambase::group_count(static_cast<u32>(n), group_size);
                const u32 corrected_len = group_cnt * group_size;

                sham::kernel_call_hndl(
                    shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
                    sham::MultiRef{xyz.get_buf()},
                    sham::MultiRef{axyz_ext.get_buf()},
                    static_cast<u32>(n),
                    [corrected_len, group_size, G, gpart_mass, gravitational_softening](
                        u32 Npart, const Tvec *__restrict xyz, Tvec *__restrict axyz_ext) {
                        auto range = sycl::nd_range<1>{corrected_len, group_size};

                        return [=](sycl::handler &cgh) {
                            using vec4 = sycl::vec<Tscal, 4>;

                            auto position_scratch
                                = sycl::local_accessor<vec4>{sycl::range<1>{group_size}, cgh};

                            cgh.parallel_for(range, [=](sycl::nd_item<1> tid) {
                                const u64 global_id = tid.get_global_linear_id();
                                const u64 local_id  = tid.get_local_linear_id();

                                Tvec force{0.0f};

                                const Tvec my_particle
                                    = (global_id < Npart) ? xyz[global_id] : Tvec{0.0f};

                                for (u32 offset = 0; offset < Npart; offset += group_size) {

                                    if (offset + local_id < Npart) {
                                        position_scratch[local_id]
                                            = vec4{xyz[offset + local_id], gpart_mass};
                                    } else {
                                        position_scratch[local_id] = vec4{0.0f};
                                    }

                                    sycl::group_barrier(tid.get_group());

                                    for (u32 i = 0; i < group_size; ++i) {
                                        const Tvec p
                                            = position_scratch[i].template swizzle<0, 1, 2>();
                                        const Tvec R = p - my_particle;

                                        const Tscal r_inv = sycl::rsqrt(
                                            R.x() * R.x() + R.y() * R.y() + R.z() * R.z()
                                            + gravitational_softening);

                                        if (global_id != offset + i) {
                                            force += position_scratch[i].w() * r_inv * r_inv * r_inv
                                                     * R;
                                        }
                                    }

                                    sycl::group_barrier(tid.get_group());
                                }

                                if (global_id < Npart) {
                                    axyz_ext[global_id] += force * G;
                                }
                            });
                        };
                    });
            }
        });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SGDirectPlummer<f64_3>;

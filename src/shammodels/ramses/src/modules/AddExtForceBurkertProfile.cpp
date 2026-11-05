// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddExtForceBurkertProfile.cpp
 * @author Jeremy Fensch (jeremy.fensch@ens-lyon.fr) from a version from Noé Brucy
 (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula for the Burkert Profile.
 */

#include "shambase/constants.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/AddExtForceBurkertProfile.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void AddExtForceBurkertProfile<Tvec>::_impl_evaluate_internal() {
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext.check_sizes(edges.sizes.indexes);

        auto &coordinates_spans = edges.spans_coordinates.get_spans();
        auto &a_ext_spans       = edges.spans_a_ext.get_spans();

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{coordinates_spans},
            sham::DDMultiRef{a_ext_spans},
            cell_counts,
            [rho_0 = rho_0, r_0 = r_0, pos = pos, const_G = const_G](
                u32 i, const Tvec *__restrict cell_coor, Tvec *__restrict a_ext) {
                Tvec cell_pos = cell_coor[i];

                // compute vector from the point mass to the cell
                Tvec r_vec = cell_pos - pos;

                Tvec r_to_r_0 = r_vec / r_0;

                // compute distance to the point mass
                Tscal r_to_r_0_squared = 0;
                Tscal r_squared        = 0;
                for (u32 d = 0; d < dim; d++) {
                    r_to_r_0_squared += r_to_r_0[d] * r_to_r_0[d];
                }

                Tscal inv_r_to_r_0           = sycl::rsqrt(r_to_r_0_squared);
                Tscal one_over_r_to_r_0_cube = inv_r_to_r_0 * inv_r_to_r_0 * inv_r_to_r_0;

                // compute gravitational acceleration using softened point mass potential
                a_ext[i] -= const_G * shambase::constants::pi<Tscal>
                            * rho_0 * one_over_r_to_r_0_cube * r_vec
                            * (sycl::log(1 + r_to_r_0_squared) + 2 * sycl::log(1 + r_to_r_0)
                               - 2 * sycl::atan(r_to_r_0));
            });
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::AddExtForceBurkertProfile<f64_3>;

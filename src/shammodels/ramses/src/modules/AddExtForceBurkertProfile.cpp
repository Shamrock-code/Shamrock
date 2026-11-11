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

        const Tscal const_G = edges.constant_G.value;

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
            [rho_0 = rho_0, r_0 = r_0, profile_center = pos, const_G = const_G](
                u32 i, const Tvec *__restrict cell_coor, Tvec *__restrict a_ext) {
                Tvec cell_pos = cell_coor[i];

                // compute vector from the profile center to the cell
                Tvec r_vec = cell_pos - profile_center;

                Tscal r_squared = 0;
                for (u32 d = 0; d < dim; d++) {
                    r_squared += r_vec[d] * r_vec[d];
                }
                Tscal r            = sycl::sqrt(r_squared);
                Tscal r_norm       = r / r_0;
                Tscal logatan_term = sycl::log(1 + r_norm * r_norm) + 2 * sycl::log(1 + r_norm)
                                     - 2 * sycl::atan(r_norm);

                a_ext[i] -= const_G * shambase::constants::pi<Tscal>
                            * rho_0 / (r_norm * r_norm * r_norm) * logatan_term * r_vec;
            });
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::AddExtForceBurkertProfile<f64_3>;

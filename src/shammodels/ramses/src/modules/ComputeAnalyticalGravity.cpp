// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeAnalyticalGravity.cpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula.
 */

#include "shammodels/ramses/modules/ComputeAnalyticalGravity.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeComputeAnalyticalGravity<Tvec>::_impl_evaluate_internal() {
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_gravitational_force.ensure_sizes(edges.sizes.indexes);

        auto &coordinates_spans         = edges.spans_coordinates.get_spans();
        auto &gravitational_force_spans = edges.spans_gravitational_force.get_spans();

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        if (config.analytical_gravity_mode == POINTMASS) {

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{coordinates_spans},
                sham::DDMultiRef{gravitational_force_spans},
                cell_counts,
                [config = this->config, block_size = this->block_size](
                    u32 i, const Tvec *__restrict cell_coor, Tvec *__restrict gravitational_force) {
                    u32 block_id  = i / block_size;
                    Tvec cell_pos = cell_coor[i];

                    // compute vector from the point mass to the cell
                    Tvec r_vec = cell_pos - config.point_mass_pos;

                    // compute distance to the point mass
                    Tscal r_squared = 0;
                    for (u32 d = 0; d < dim; d++) {
                        r_squared += r_vec[d] * r_vec[d];
                    }

                    Tscal one_over_r_cube_soft = pow(r_squared + config.epsilon_softening, -1.5f);

                    // compute gravitational acceleration using softened point mass potential
                    gravitational_force[i] = -config.point_mass_GM * r_vec * one_over_r_cube_soft;
                });
        }
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeAnalyticalGravity<f64_3>;

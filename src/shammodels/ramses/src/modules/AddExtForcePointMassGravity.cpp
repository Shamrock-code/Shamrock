// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddExtForcePointMassGravity.cpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula.
 */

#include "shammodels/ramses/modules/AddExtForcePointMassGravity.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void AddExtForcePointMassGravity<Tvec>::_impl_evaluate_internal() {
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext.check_sizes(edges.sizes.indexes);

        auto &coordinates_spans = edges.spans_coordinates.get_spans();
        auto &a_ext_spans       = edges.spans_a_ext.get_spans();

        const Tscal gmass = mass * edges.constant_G.value;

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
            [pos = pos, gmass = gmass, epsilon_soft = epsilon_soft](
                u32 i, const Tvec *__restrict cell_coor, Tvec *__restrict a_ext) {
                Tvec cell_pos = cell_coor[i];

                // compute vector from the point mass to the cell
                Tvec r_vec = cell_pos - pos;

                // compute distance to the point mass
                Tscal r_squared = sycl::dot(r_vec, r_vec);

                Tscal inv_r_soft           = sycl::rsqrt(r_squared + epsilon_soft * epsilon_soft);
                Tscal one_over_r_cube_soft = inv_r_soft * inv_r_soft * inv_r_soft;

                // compute gravitational acceleration using softened point mass potential
                a_ext[i] -= gmass * r_vec * one_over_r_cube_soft;
            });
    }

    template<class Tvec>
    std::string AddExtForcePointMassGravity<Tvec>::_impl_get_tex() const {
        auto coords = get_ro_edge_base(1).get_tex_symbol();
        auto a_ext  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Add the gravity of a softened point mass to the external forces acceleration

        \begin{align}
        {{a_ext}}_i &= {{a_ext}}_i - \frac{G M \left({{coords}}_i - \mathbf{r}_0\right)}
            {\left(\left|{{coords}}_i - \mathbf{r}_0\right|^2 + \epsilon^2\right)^{3/2}}
        \end{align}
    )tex";

        shambase::replace_all(tex, "{a_ext}", a_ext);
        shambase::replace_all(tex, "{coords}", coords);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::AddExtForcePointMassGravity<f64_3>;

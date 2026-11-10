// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyBetaCooling.cpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Apply beta-cooling (cooling time proportional to the rotation time)
 */

#include "shammodels/ramses/modules/ApplyBetaCooling.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyBetaCooling<Tvec>::_impl_evaluate_internal() {
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        Tscal dt = edges.dt.value;

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        auto &coordinates_spans = edges.spans_coordinates.get_spans();
        auto &rho_spans         = edges.spans_rho.get_spans();
        auto &rhov_spans        = edges.spans_rhov.get_spans();
        auto &rhoe_spans        = edges.spans_rhoe.get_spans();

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{coordinates_spans, rho_spans, rhov_spans},
            sham::DDMultiRef{rhoe_spans},
            cell_counts,
            [center = center, beta = beta, dt = dt](
                u32 i,
                const Tvec *__restrict cell_coor,
                const Tscal *__restrict rho,
                const Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                Tvec cell_pos = cell_coor[i];

                // compute vector from the centerto the cell
                Tvec r_vec = cell_pos - center;
                Tscal x    = r_vec[0];
                Tscal y    = r_vec[1];
                Tscal vx   = rhov[i][0] / rho[i];
                Tscal vy   = rhov[i][1] / rho[i];

                // compute distance to the point mass
                Tscal r_squared = sycl::max(x * x + y * y, 1e-20);

                Tscal omega = (-y * vx + x * vy) / r_squared;

                // compute old thermal energy
                Tscal ekin = 0.5 * sycl::dot(rhov[i], rhov[i]) / rho[i];
                Tscal eint = rhoe[i] - ekin;

                // compute new thermal energy
                eint *= sycl::exp(-dt * sycl::fabs(omega) / beta);

                rhoe[i] = ekin + eint;
            });
    }

    template<class Tvec>
    std::string ApplyBetaCooling<Tvec>::_impl_get_tex() const {
        auto rhoe = get_rw_edge_base(0).get_tex_symbol();
        auto dt   = get_ro_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Apply $\beta$-cooling

        \begin{align}
        {{rhoe}}_i &= {{rhoe}}_i - e_\mathrm{int,i} + e_\mathrm{int,i} \exp\left(-\frac{ {{dt}} \Omega } {\beta} \right)
        \end{align}
    )tex";

        shambase::replace_all(tex, "{rhoe}", rhoe);
        shambase::replace_all(tex, "{dt}", dt);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyBetaCooling<f64_3>;

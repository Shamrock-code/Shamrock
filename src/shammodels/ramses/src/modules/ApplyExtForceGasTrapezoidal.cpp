// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyExtForceGasTrapezoidal.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the external forces accelerations at t^n and t^{n+1} to the gas
 * conservative state (trapezoidal rule)
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyExtForceGasTrapezoidal.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyExtForceGasTrapezoidal<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext_n.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext_np1.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        Tscal dt       = edges.dt.value;
        Tscal dt_over2 = dt / 2;

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_rho.get_spans(),
                edges.spans_dt_rho.get_spans(),
                edges.spans_dt_rhov.get_spans(),
                edges.spans_a_ext_n.get_spans(),
                edges.spans_a_ext_np1.get_spans()},
            sham::DDMultiRef{edges.spans_rhov.get_spans(), edges.spans_rhoe.get_spans()},
            cell_counts,
            [dt, dt_over2](
                u32 i,
                const Tscal *__restrict rho,
                const Tscal *__restrict dt_rho,
                const Tvec *__restrict dt_rhov,
                const Tvec *__restrict a_ext_n,
                const Tvec *__restrict a_ext_np1,
                Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                Tscal rho_n = rho[i];
                Tvec rhov_n = rhov[i];

                // density at t^{n+1}
                Tscal rho_np1 = rho_n + dt * dt_rho[i];

                Tvec kick = dt_over2 * (rho_n * a_ext_n[i] + rho_np1 * a_ext_np1[i]);

                // momentum at t^{n+1}, including the kick of the external forces
                Tvec rhov_np1 = rhov_n + dt * dt_rhov[i] + kick;

                rhov[i] += kick;
                rhoe[i] += dt_over2
                           * (sycl::dot(a_ext_n[i], rhov_n) + sycl::dot(a_ext_np1[i], rhov_np1));
            });
    }

    template<class Tvec>
    std::string ApplyExtForceGasTrapezoidal<Tvec>::_impl_get_tex() const {
        auto dt        = get_ro_edge_base(0).get_tex_symbol();
        auto rho       = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rho    = get_ro_edge_base(3).get_tex_symbol();
        auto dt_rhov   = get_ro_edge_base(4).get_tex_symbol();
        auto a_ext_n   = get_ro_edge_base(5).get_tex_symbol();
        auto a_ext_np1 = get_ro_edge_base(6).get_tex_symbol();
        auto rhov      = get_rw_edge_base(0).get_tex_symbol();
        auto rhoe      = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Apply external forces to the gas (trapezoidal rule)

            \begin{align}
            \rho^{n+1}_i &= {rho}_i + {dt} \, {dt_rho}_i \
            \Delta (\rho \mathbf{v})_i &= \frac{{dt}}{2} \left( {rho}_i {a_ext_n}_i
                + \rho^{n+1}_i {a_ext_np1}_i \right) \
            (\rho \mathbf{v})^{n+1}_i &= {rhov}_i + {dt} \, {dt_rhov}_i
                + \Delta (\rho \mathbf{v})_i \
            {rhoe}_i &= {rhoe}_i + \frac{{dt}}{2} \left( {a_ext_n}_i \cdot {rhov}_i
                + {a_ext_np1}_i \cdot (\rho \mathbf{v})^{n+1}_i \right) \
            {rhov}_i &= {rhov}_i + \Delta (\rho \mathbf{v})_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt_rhov}", dt_rhov);
        shambase::replace_all(tex, "{a_ext_np1}", a_ext_np1);
        shambase::replace_all(tex, "{a_ext_n}", a_ext_n);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyExtForceGasTrapezoidal<f64_3>;

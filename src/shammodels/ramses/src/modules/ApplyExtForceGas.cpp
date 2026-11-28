// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyExtForceGas.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyExtForceGas.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyExtForceGas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext.check_sizes(edges.sizes.indexes);
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
                edges.spans_a_ext.get_spans()},
            sham::DDMultiRef{edges.spans_rhov.get_spans(), edges.spans_rhoe.get_spans()},
            cell_counts,
            [dt, dt_over2](
                u32 i,
                const Tscal *__restrict rho,
                const Tscal *__restrict dt_rho,
                const Tvec *__restrict dt_rhov,
                const Tvec *__restrict a_ext,
                Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                // time centered density & momentum at t^{n+1/2}
                Tscal rho_half = rho[i] + dt_over2 * dt_rho[i];
                Tvec rhov_half = rhov[i] + dt_over2 * dt_rhov[i];

                Tvec a = a_ext[i];

                // TODO: to be confirmed, rhov_half does not include the kick of the external
                // forces, so for a free fall the internal energy should drift by
                // ~rho a^2 dt^2 / 2 per step (first order). Check if the energy update should
                // instead use the momentum including the kick (as in RAMSES, which recomputes
                // the kinetic energy after the kick).
                rhov[i] += dt * rho_half * a;
                rhoe[i] += dt * sycl::dot(a, rhov_half);
            });
    }

    template<class Tvec>
    std::string ApplyExtForceGas<Tvec>::_impl_get_tex() const {
        auto dt      = get_ro_edge_base(0).get_tex_symbol();
        auto rho     = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rho  = get_ro_edge_base(3).get_tex_symbol();
        auto dt_rhov = get_ro_edge_base(4).get_tex_symbol();
        auto a_ext   = get_ro_edge_base(5).get_tex_symbol();
        auto rhov    = get_rw_edge_base(0).get_tex_symbol();
        auto rhoe    = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Apply external forces to the gas

            \begin{align}
            \rho^{n+1/2}_i &= {rho}_i + \frac{{dt}}{2} \, {dt_rho}_i \
            (\rho \mathbf{v})^{n+1/2}_i &= {rhov}_i + \frac{{dt}}{2} \, {dt_rhov}_i \
            {rhov}_i &= {rhov}_i + {dt} \, \rho^{n+1/2}_i {a_ext}_i \
            {rhoe}_i &= {rhoe}_i + {dt} \, {a_ext}_i \cdot (\rho \mathbf{v})^{n+1/2}_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt_rhov}", dt_rhov);
        shambase::replace_all(tex, "{a_ext}", a_ext);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyExtForceGas<f64_3>;

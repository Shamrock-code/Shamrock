// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ForwardEulerGas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ForwardEulerGas.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ForwardEulerGas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_dt_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhoe.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        Tscal dt = edges.dt.value;

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_dt_rho.get_spans(),
                edges.spans_dt_rhov.get_spans(),
                edges.spans_dt_rhoe.get_spans()},
            sham::DDMultiRef{
                edges.spans_rho.get_spans(),
                edges.spans_rhov.get_spans(),
                edges.spans_rhoe.get_spans()},
            cell_counts,
            [dt](
                u32 i,
                const Tscal *__restrict dt_rho,
                const Tvec *__restrict dt_rhov,
                const Tscal *__restrict dt_rhoe,
                Tscal *__restrict rho,
                Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                rho[i] += dt * dt_rho[i];
                rhov[i] += dt * dt_rhov[i];
                rhoe[i] += dt * dt_rhoe[i];
            });
    }

    template<class Tvec>
    std::string ForwardEulerGas<Tvec>::_impl_get_tex() const {
        auto dt      = get_ro_edge_base(0).get_tex_symbol();
        auto dt_rho  = get_ro_edge_base(2).get_tex_symbol();
        auto dt_rhov = get_ro_edge_base(3).get_tex_symbol();
        auto dt_rhoe = get_ro_edge_base(4).get_tex_symbol();
        auto rho     = get_rw_edge_base(0).get_tex_symbol();
        auto rhov    = get_rw_edge_base(1).get_tex_symbol();
        auto rhoe    = get_rw_edge_base(2).get_tex_symbol();

        std::string tex = R"tex(
            Forward Euler integration of the gas

            \begin{align}
            {rho}_i &= {rho}_i + {dt} \, {dt_rho}_i \
            {rhov}_i &= {rhov}_i + {dt} \, {dt_rhov}_i \
            {rhoe}_i &= {rhoe}_i + {dt} \, {dt_rhoe}_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt_rhov}", dt_rhov);
        shambase::replace_all(tex, "{dt_rhoe}", dt_rhoe);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ForwardEulerGas<f64_3>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SGSPHCorrection.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/self_gravity/SGSPHCorrection.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammath/sphkernels.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void SGSPHCorrection<Tvec, SPHKernel>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        const Tscal G          = edges.constant_G.data;
        const Tscal gpart_mass = edges.gpart_mass.data;

        edges.add_to_force.ensure_sizes(edges.part_counts.indexes);

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.xyz.get_spans(),
                edges.hpart.get_spans(),
                edges.omega.get_spans(),
                edges.xi.get_spans(),
                edges.neigh_cache.neigh_cache},
            sham::DDMultiRef{edges.add_to_force.get_spans()},
            edges.part_counts.indexes,
            [half_G = G / 2, gpart_mass](
                u32 id_a,
                const Tvec *r,
                const Tscal *hpart,
                const Tscal *omega,
                const Tscal *xi,
                const auto ploop_ptrs,
                Tvec *add_to_force) {
                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                Tvec sum = {};

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    Tvec dr        = r[id_a] - r[id_b];
                    Tscal rab2     = sycl::dot(dr, dr);
                    Tscal rab      = sycl::sqrt(rab2);
                    Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                    Tscal omega_a = omega[id_a];
                    Tscal omega_b = omega[id_b];
                    Tscal xi_a    = xi[id_a];
                    Tscal xi_b    = xi[id_b];

                    Tscal h_a = hpart[id_a];
                    Tscal h_b = hpart[id_b];

                    Tscal F_ab_a = SPHKernel<Tscal>::dW_3d(rab, h_a);
                    Tscal F_ab_b = SPHKernel<Tscal>::dW_3d(rab, h_b);

                    Tvec nabla_Wab_ha = r_ab_unit * F_ab_a;
                    Tvec nabla_Wab_hb = r_ab_unit * F_ab_b;

                    sum += gpart_mass
                           * ((xi_a / omega_a) * nabla_Wab_ha + (xi_b / omega_b) * nabla_Wab_hb);
                });

                add_to_force[id_a] = -half_G * sum;
            });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::M4>;
template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::M6>;
template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::M8>;
template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::C2>;
template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::C4>;
template class shammodels::sph::modules::SGSPHCorrection<f64_3, shammath::C6>;

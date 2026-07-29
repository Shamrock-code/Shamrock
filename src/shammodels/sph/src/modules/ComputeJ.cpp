// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeJ.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/math/mhd.hpp"
#include "shammodels/sph/modules/ComputeJ.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include <fmt/ranges.h>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeComputeJ<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();
    logger::raw_ln("xinside compute J");
    logger::raw_ln("1111 c = ", c);
    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    logger::raw_ln("before ensure size");
    edges.J.ensure_sizes(edges.part_counts.indexes);
    logger::raw_ln("after ensure size");
    Tscal _pi = shambase::constants::pi<Tscal>;

    edges.part_counts.indexes.for_each([&](u64 id_patch, u32 count) {
        fmt::print("patch {} has {} particles\n", id_patch, count);
    });

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.neigh_cache.neigh_cache,
            edges.omega.get_spans(),
            edges.B_on_rho.get_spans()},
        sham::DDMultiRef{edges.J.get_spans()},
        edges.part_counts.indexes,
        [part_mass = this->part_mass,
         mu_0      = this->mu_0,
         c         = this->c,
         _pi       = _pi,
         Rkern     = kernel_radius](
            u32 id_a,
            const Tvec *r,
            const Tscal *hpart,
            const auto ploop_ptrs,
            const Tscal *omega,
            const Tvec *B_on_rho,
            Tvec *J) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            using namespace shamrock::sph;
            using namespace shamrock::sph::mhd;

            Tvec xyz_a = r[id_a]; // could be recovered from lambda

            Tscal h_a = hpart[id_a];

            Tscal rho_a    = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
            Tscal rho_a_sq = rho_a * rho_a;

            Tvec B_a         = B_on_rho[id_a] * rho_a;
            Tscal omega_a    = omega[id_a];
            Tscal sub_fact_a = rho_a * omega_a;

            Tscal part_omega_sum = 0;
            Tvec J_sum{0, 0, 0};

            constexpr Tscal Rker2 = SPHKernel<Tscal>::Rkern * SPHKernel<Tscal>::Rkern;

            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                Tvec dr    = xyz_a - r[id_b];
                Tscal rab2 = sycl::dot(dr, dr);
                Tscal h_b  = hpart[id_b];

                if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                    return;
                }

                Tscal rab   = sycl::sqrt(rab2);
                Tscal rho_b = rho_h(part_mass, h_b, SPHKernel<Tscal>::hfactd);
                // if (id_a == 0) {
                //     logger::raw_ln("@@@@@@ h_b", h_b);
                //     logger::raw_ln("idb", id_b);
                //     logger::raw_ln("@@@@@@ xyz_b", r[id_b]);
                // }
                Tvec B_b = B_on_rho[id_b] * rho_b;

                Tscal Fab_a       = SPHKernel<Tscal>::dW_3d(rab, h_a);
                Tvec r_ab_unit    = dr * sham::inv_sat_positive(rab);
                Tvec nabla_Wab_ha = r_ab_unit * Fab_a;

                J_sum += shamrock::sph::mhd::MagCurrentJ_sum(
                    part_mass, B_a, B_b, nabla_Wab_ha, sub_fact_a, mu_0);
            });

            J[id_a] = J_sum; //* 4 * _pi / c;
        });
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeComputeJ<Tvec, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::NodeComputeJ<f64_3, M4>;
template class shammodels::sph::modules::NodeComputeJ<f64_3, M6>;
template class shammodels::sph::modules::NodeComputeJ<f64_3, M8>;

template class shammodels::sph::modules::NodeComputeJ<f64_3, C2>;
template class shammodels::sph::modules::NodeComputeJ<f64_3, C4>;
template class shammodels::sph::modules::NodeComputeJ<f64_3, C6>;

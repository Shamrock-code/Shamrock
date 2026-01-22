// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeOmega.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/modules/ComputeOmega.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeComputeOmega<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    edges.omega.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(), edges.hpart.get_spans(), edges.neigh_cache.neigh_cache},
        sham::DDMultiRef{edges.omega.get_spans()},
        edges.part_counts.indexes,
        [part_mass = this->part_mass, Rkern = kernel_radius](
            u32 id_a, const Tvec *r, const Tscal *hpart, const auto ploop_ptrs, Tscal *omega) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            Tvec xyz_a = r[id_a]; // could be recovered from lambda

            Tscal h_a  = hpart[id_a];
            Tscal dint = h_a * h_a * Rkern * Rkern;

            Tscal rho_sum        = 0;
            Tscal part_omega_sum = 0;

            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                Tvec dr    = xyz_a - r[id_b];
                Tscal rab2 = sycl::dot(dr, dr);

                if (rab2 > dint) {
                    return;
                }

                Tscal rab = sycl::sqrt(rab2);

                rho_sum += part_mass * SPHKernel<Tscal>::W_3d(rab, h_a);
                part_omega_sum += part_mass * SPHKernel<Tscal>::dhW_3d(rab, h_a);
            });

            using namespace shamrock::sph;

            Tscal rho_ha  = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
            Tscal omega_a = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
            omega[id_a]   = omega_a;
        });
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeComputeOmega<Tvec, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

template<class T>
void shammodels::sph::modules::SetWhenMask<T>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    edges.mask.check_sizes(edges.part_counts.indexes);
    edges.field_to_set.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{edges.mask.get_spans()},
        sham::DDMultiRef{edges.field_to_set.get_spans()},
        edges.part_counts.indexes,
        [val_to_set = this->val_to_set](u32 id, const u32 *mask, T *field_to_set) {
            if (mask[id] == 1) {
                field_to_set[id] = val_to_set;
            }
        });
}

template<class T>
std::string shammodels::sph::modules::SetWhenMask<T>::_impl_get_tex() const {
    return "TODO";
}

template class shammodels::sph::modules::SetWhenMask<f64>;

using namespace shammath;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M4>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M6>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M8>;

template class shammodels::sph::modules::NodeComputeOmega<f64_3, C2>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, C4>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, C6>;

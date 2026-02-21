// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsCD10.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/NodeUpdateDerivsCD10.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shamrock/patch/PatchDataField.hpp"

template<class Tvec, template<class> class SPHKernel>
struct KernelUpdateDerivsCD10 {
    using Tscal                   = shambase::VecComponent<Tvec>;
    using Kernel                  = SPHKernel<Tscal>;
    static constexpr Tscal hfactd = Kernel::hfactd;
    static constexpr Tscal Rkern  = Kernel::Rkern;
    static constexpr Tscal Rker2  = Rkern * Rkern;

    Tscal pmass;
    Tscal alpha_u;
    Tscal beta_AV;

    inline void operator()(
        unsigned int id_a,
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict uint,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict cs,
        const Tscal *__restrict alpha_AV,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        Tvec *__restrict axyz,
        Tscal *__restrict duint) const {

        using namespace shamrock::sph;

        shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

        Tvec xyz_a    = xyz[id_a];
        Tscal h_a     = hpart[id_a];
        Tvec vxyz_a   = vxyz[id_a];
        Tscal u_a     = uint[id_a];
        Tscal omega_a = omega[id_a];
        Tscal P_a     = pressure[id_a];
        Tscal cs_a    = cs[id_a];
        Tscal alpha_a = alpha_AV[id_a];

        Tscal rho_a     = rho_h(pmass, h_a, hfactd);
        Tscal rho_a_sq  = rho_a * rho_a;
        Tscal rho_a_inv = 1. / rho_a;

        Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

        Tvec force_pressure  = Tvec{0, 0, 0};
        Tscal tmpdU_pressure = Tscal{0};

        particle_looper.for_each_object(id_a, [&](u32 id_b) {
            Tvec dr    = xyz_a - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);
            Tscal h_b  = hpart[id_b];

            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                return;
            }

            Tvec vxyz_b         = vxyz[id_b];
            const Tscal u_b     = uint[id_b];
            Tscal P_b           = pressure[id_b];
            Tscal omega_b       = omega[id_b];
            const Tscal alpha_b = alpha_AV[id_b];
            Tscal cs_b          = cs[id_b];

            Tscal rab = sycl::sqrt(rab2);

            Tscal rho_b = rho_h(pmass, h_b, hfactd);

            Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            Tvec v_ab = vxyz_a - vxyz_b;

            Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
            Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

            Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
            Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

            Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

            Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
            Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

            add_to_derivs_sph_artif_visco_cond(
                pmass,
                rho_a_sq,
                omega_a_rho_a_inv,
                rho_a_inv,
                rho_b,
                omega_a,
                omega_b,
                Fab_a,
                Fab_b,
                u_a,
                u_b,
                P_a,
                P_b,
                alpha_u,
                v_ab,
                r_ab_unit,
                vsig_u,
                qa_ab,
                qb_ab,

                force_pressure,
                tmpdU_pressure);
        });

        axyz[id_a]  = force_pressure;
        duint[id_a] = tmpdU_pressure;
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsCD10<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts_with_ghost = edges.part_counts_with_ghost.indexes;
    auto &part_counts            = edges.part_counts.indexes;

    // check that all input edges have the particles with ghosts zones
    edges.xyz.check_sizes(part_counts_with_ghost);
    edges.hpart.check_sizes(part_counts_with_ghost);
    edges.vxyz.check_sizes(part_counts_with_ghost);
    edges.uint.check_sizes(part_counts_with_ghost);
    edges.omega.check_sizes(part_counts_with_ghost);
    edges.pressure.check_sizes(part_counts_with_ghost);
    edges.cs.check_sizes(part_counts_with_ghost);
    edges.alpha_AV.check_sizes(part_counts_with_ghost);

    // ensure that the output edges are of size part_counts (output without ghosts zones)
    edges.axyz.ensure_sizes(part_counts);
    edges.duint.ensure_sizes(part_counts);

    const Tscal pmass   = edges.gpart_mass.value;
    const Tscal alpha_u = edges.alpha_u.value;
    const Tscal beta_AV = edges.beta_AV.value;

    using ComputeKernel = KernelUpdateDerivsCD10<Tvec, SPHKernel>;

    // call the kernel for each patches with part_counts.get(id_patch) threads of patch id_patch
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.vxyz.get_spans(),
            edges.uint.get_spans(),
            edges.omega.get_spans(),
            edges.pressure.get_spans(),
            edges.cs.get_spans(),
            edges.alpha_AV.get_spans(),
            edges.neigh_cache},
        sham::DDMultiRef{edges.axyz.get_spans(), edges.duint.get_spans()},
        part_counts,
        ComputeKernel{pmass, alpha_u, beta_AV});
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsCD10<f64_3, C6>;

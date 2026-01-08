// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SRPrimitiveRecovery.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Primitive recovery implementation for SR-GSPH
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/physics/sr/SRPrimitiveRecovery.hpp"
#include "shammodels/gsph/physics/sr/recovery/NewtonRaphson.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::physics::sr {

    template<class Tvec, template<class> class SPHKernel>
    void SRPrimitiveRecovery<Tvec, SPHKernel>::init_conserved(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        if (storage.sr_initialized) {
            return;
        }

        // Ensure fields exist with correct sizes
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
            = std::make_shared<shamrock::solvergraph::Indexes<u32>>("sr_sizes", "N");
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        storage.S_momentum->ensure_sizes(sizes->indexes);
        storage.e_energy->ensure_sizes(sizes->indexes);
        storage.dS_momentum->ensure_sizes(sizes->indexes);
        storage.de_energy->ensure_sizes(sizes->indexes);

        // Zero-initialize dS and de
        shamrock::solvergraph::Field<Tvec> &dS_field  = *storage.dS_momentum;
        shamrock::solvergraph::Field<Tscal> &de_field = *storage.de_energy;

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{buf_dS, buf_de},
                cnt,
                [](u32 i, Tvec *dS, Tscal *de) {
                    dS[i] = Tvec{0, 0, 0};
                    de[i] = Tscal{0};
                });
        });

        // Compute S and e from initial primitives (v, P)
        // Kitajima volume-based approach: N = pmass × (hfact/h)³
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 ipmass          = pdl.get_field_idx<Tscal>("pmass");

        const Tscal c_speed   = config.c_speed;
        const Tscal gamma_eos = config.get_eos_gamma();
        const Tscal hfact     = SPHKernel<Tscal>::hfactd;

        shamrock::solvergraph::Field<Tvec> &S_field  = *storage.S_momentum;
        shamrock::solvergraph::Field<Tscal> &e_field = *storage.e_energy;
        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_vxyz   = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tscal> &buf_h     = pdat.get_field_buf_ref<Tscal>(ihpart);
            sham::DeviceBuffer<Tscal> &buf_pmass = pdat.get_field_buf_ref<Tscal>(ipmass);
            sham::DeviceBuffer<Tvec> &buf_S      = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e     = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_P     = pressure_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_vxyz, buf_h, buf_pmass, buf_P},
                sham::MultiRef{buf_S, buf_e},
                cnt,
                [c_speed, gamma_eos, hfact](
                    u32 i,
                    const Tvec *vxyz,
                    const Tscal *h,
                    const Tscal *pmass,
                    const Tscal *P,
                    Tvec *S,
                    Tscal *e) {
                    const Tscal c2 = c_speed * c_speed;

                    // Compute Lorentz factor
                    const Tscal v2        = sycl::dot(vxyz[i], vxyz[i]) / c2;
                    const Tscal gamma_lor = Tscal{1} / sycl::sqrt(Tscal{1} - v2);

                    // Kitajima volume-based N: N = pmass × (hfact/h)³
                    const Tscal h_ratio = hfact / h[i];
                    const Tscal N_lab   = pmass[i] * h_ratio * h_ratio * h_ratio;

                    // Lab-frame N to rest-frame n
                    const Tscal n = N_lab / gamma_lor;

                    // Internal energy per unit mass (from EOS: P = (γ-1) × n × u)
                    const Tscal u_int = P[i] / ((gamma_eos - Tscal{1}) * n);

                    // Specific enthalpy: H = 1 + u/c² + P/(nc²)
                    const Tscal H = Tscal{1} + u_int / c2 + P[i] / (n * c2);

                    // Conserved variables
                    S[i] = vxyz[i] * (gamma_lor * H);
                    e[i] = gamma_lor * H - P[i] / (N_lab * c2);
                });
        });

        // dS and de were already zero-initialized above, so predictor on next timestep
        // won't apply stale values from forces computed before CFL was established

        storage.sr_initialized = true;

        if (shamcomm::world_rank() == 0) {
            shamcomm::logs::raw_ln(
                "SR-GSPH: Initialized conserved variables (S, e) from primitives");
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPrimitiveRecovery<Tvec, SPHKernel>::recover(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};

        NewtonRaphsonConfig recovery_cfg;
        recovery_cfg.tol      = config.sr_tol;
        recovery_cfg.max_iter = config.sr_max_iter;

        recover_newton_raphson(storage, config, scheduler, recovery_cfg);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPrimitiveRecovery<Tvec, SPHKernel>::recover_newton_raphson(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const NewtonRaphsonConfig &recovery_config) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;
        namespace sr_math = shammodels::gsph::physics::sr::recovery;

        // Ensure output fields have correct sizes
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
            = std::make_shared<shamrock::solvergraph::Indexes<u32>>("recover_sizes", "N");
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });
        storage.density->ensure_sizes(sizes->indexes);
        storage.pressure->ensure_sizes(sizes->indexes);

        // Kitajima volume-based approach: N = pmass × (hfact/h)³
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 ipmass          = pdl.get_field_idx<Tscal>("pmass");
        const bool has_uint       = config.has_field_uint();
        const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

        const Tscal c_speed   = config.c_speed;
        const Tscal gamma_eos = config.get_eos_gamma();
        const Tscal hfact     = SPHKernel<Tscal>::hfactd;

        shamrock::solvergraph::Field<Tvec> &S_field  = *storage.S_momentum;
        shamrock::solvergraph::Field<Tscal> &e_field = *storage.e_energy;
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);
        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_vxyz   = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tscal> &buf_h     = pdat.get_field_buf_ref<Tscal>(ihpart);
            sham::DeviceBuffer<Tscal> &buf_pmass = pdat.get_field_buf_ref<Tscal>(ipmass);
            sham::DeviceBuffer<Tvec> &buf_S      = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e     = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_rho   = density_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_P     = pressure_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            // Kitajima: compute N from h, then recover primitives
            // Also write recovered rest-frame density n to density_field for output
            if (has_uint) {
                sham::DeviceBuffer<Tscal> &buf_uint = pdat.get_field_buf_ref<Tscal>(iuint);

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{buf_S, buf_e, buf_h, buf_pmass},
                    sham::MultiRef{buf_vxyz, buf_P, buf_uint, buf_rho},
                    cnt,
                    [c_speed, gamma_eos, hfact](
                        u32 i,
                        const Tvec *S,
                        const Tscal *e,
                        const Tscal *h,
                        const Tscal *pmass,
                        Tvec *vxyz,
                        Tscal *P,
                        Tscal *uint_out,
                        Tscal *rho_out) {
                        const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                        const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                        // Kitajima volume-based N: N = pmass × (hfact/h)³
                        const Tscal h_ratio = hfact / h[i];
                        const Tscal N       = pmass[i] * h_ratio * h_ratio * h_ratio;

                        sr_math::Result<Tscal> prim
                            = sr_math::recover<Tscal>(S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                        vxyz[i] = S_dir * prim.vel_normal * c_speed;
                        P[i]    = prim.pressure;

                        // Output rest-frame density n for VTK
                        rho_out[i] = prim.density;

                        // uint for compatibility: P = (γ-1) * n * u => u = P / ((γ-1) * n)
                        uint_out[i] = P[i] / ((gamma_eos - Tscal{1}) * prim.density);
                    });
            } else {
                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{buf_S, buf_e, buf_h, buf_pmass},
                    sham::MultiRef{buf_vxyz, buf_P, buf_rho},
                    cnt,
                    [c_speed, gamma_eos, hfact](
                        u32 i,
                        const Tvec *S,
                        const Tscal *e,
                        const Tscal *h,
                        const Tscal *pmass,
                        Tvec *vxyz,
                        Tscal *P,
                        Tscal *rho_out) {
                        const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                        const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                        // Kitajima volume-based N: N = pmass × (hfact/h)³
                        const Tscal h_ratio = hfact / h[i];
                        const Tscal N       = pmass[i] * h_ratio * h_ratio * h_ratio;

                        sr_math::Result<Tscal> prim
                            = sr_math::recover<Tscal>(S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                        vxyz[i] = S_dir * prim.vel_normal * c_speed;
                        P[i]    = prim.pressure;

                        // Output rest-frame density n for VTK
                        rho_out[i] = prim.density;
                    });
            }
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPrimitiveRecovery<Tvec, SPHKernel>::recover_noble_2d(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const Noble2DConfig &recovery_config) {

        shambase::throw_unimplemented("Noble 2D primitive recovery not yet implemented");
    }

    // Explicit instantiations
    using namespace shammath;

    template class SRPrimitiveRecovery<sycl::vec<double, 3>, M4>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, M6>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, M8>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, C2>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, C4>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, C6>;
    template class SRPrimitiveRecovery<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::sr

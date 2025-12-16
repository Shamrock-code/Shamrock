// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeEos.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/exception.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/ComputeEos.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tscal>
struct RhoGetterBase {
    sham::DeviceBuffer<Tscal> &buf_h;
    Tscal pmass;
    Tscal hfact;

    struct accessed {
        const Tscal *h;
        Tscal pmass;
        Tscal hfact;

        Tscal operator()(u32 i) const {
            using namespace shamrock::sph;
            return rho_h(pmass, h[i], hfact);
        }
    };

    accessed get_read_access(sham::EventList &depends_list) {
        auto h = buf_h.get_read_access(depends_list);
        return accessed{h, pmass, hfact};
    }

    void complete_event_state(sycl::event e) { buf_h.complete_event_state(e); }
};

template<class Tscal>
struct RhoGetterMonofluid {
    sham::DeviceBuffer<Tscal> &buf_h;
    sham::DeviceBuffer<Tscal> &buf_epsilon;
    u32 nvar_dust;
    Tscal pmass;
    Tscal hfact;

    struct accessed {
        const Tscal *h;
        const Tscal *buf_epsilon;
        u32 nvar_dust;
        Tscal pmass;
        Tscal hfact;

        Tscal operator()(u32 i) const {

            Tscal epsilon_sum = 0;
            for (u32 j = 0; j < nvar_dust; j++) {
                epsilon_sum += buf_epsilon[i * nvar_dust + j];
            }

            using namespace shamrock::sph;
            return (1 - epsilon_sum) * rho_h(pmass, h[i], hfact);
        }
    };

    accessed get_read_access(sham::EventList &depends_list) {
        auto h       = buf_h.get_read_access(depends_list);
        auto epsilon = buf_epsilon.get_read_access(depends_list);

        return accessed{h, epsilon, nvar_dust, pmass, hfact};
    }

    void complete_event_state(sycl::event e) { buf_h.complete_event_state(e); }
};

template<class Tvec, template<class> class SPHKernel>
template<class RhoGetGen>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos_internal(
    RhoGetGen &&rho_getter_gen) {

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 iuint_interf = ghost_layout.get_field_idx<Tscal>("uint");

    using namespace shamrock;
    using namespace shamrock::patch;

    using SolverConfigEOS                   = typename Config::EOSConfig;
    using SolverEOS_Isothermal              = typename SolverConfigEOS::Isothermal;
    using SolverEOS_Adiabatic               = typename SolverConfigEOS::Adiabatic;
    using SolverEOS_Polytropic              = typename SolverConfigEOS::Polytropic;
    using SolverEOS_LocallyIsothermal       = typename SolverConfigEOS::LocallyIsothermal;
    using SolverEOS_LocallyIsothermalLP07   = typename SolverConfigEOS::LocallyIsothermalLP07;
    using SolverEOS_LocallyIsothermalFA2014 = typename SolverConfigEOS::LocallyIsothermalFA2014;
    using SolverEOS_Tillotson               = typename SolverConfigEOS::Tillotson;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    if (SolverEOS_Isothermal *eos_config
        = std::get_if<SolverEOS_Isothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Isothermal<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            auto rho_getter = rho_getter_gen(mpdat);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [cs_cfg
                 = eos_config->cs](u32 i, auto rho, Tscal *__restrict P, Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho(i);
                    Tscal P_a   = EOS::pressure(cs_cfg, rho_a);
                    P[i]        = P_a;
                    cs[i]       = cs_cfg;
                });
        });
    } else if (
        SolverEOS_Adiabatic *eos_config
        = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Adiabatic<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [gamma = eos_config->gamma](
                    u32 i,
                    auto rho,
                    const Tscal *__restrict U,
                    Tscal *__restrict P,
                    Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho(i);
                    Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
                    Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
                    P[i]        = P_a;
                    cs[i]       = cs_a;
                });
        });

    } else if (
        SolverEOS_Polytropic *eos_config
        = std::get_if<SolverEOS_Polytropic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Polytropic<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            auto rho_getter = rho_getter_gen(mpdat);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [K = eos_config->K, gamma = eos_config->gamma](
                    u32 i, auto rho, Tscal *__restrict P, Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho(i);
                    Tscal P_a   = EOS::pressure(gamma, K, rho_a);
                    Tscal cs_a  = EOS::soundspeed(gamma, K, rho_a);
                    P[i]        = P_a;
                    cs[i]       = cs_a;
                });
        });

    } else if (
        SolverEOS_LocallyIsothermal *eos_config
        = std::get_if<SolverEOS_LocallyIsothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        u32 isoundspeed_interf = ghost_layout.get_field_idx<Tscal>("soundspeed");

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);
            sham::DeviceBuffer<Tscal> &buf_cs0 = mpdat.get_field_buf_ref<Tscal>(isoundspeed_interf);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint, buf_cs0},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [](u32 i,
                   auto rho,
                   const Tscal *__restrict U,
                   const Tscal *__restrict cs0,
                   Tscal *__restrict P,
                   Tscal *__restrict cs) {
                    using namespace shamrock::sph;

                    Tscal cs_out = cs0[i];
                    Tscal rho_a  = rho(i);

                    Tscal P_a = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[i]  = P_a;
                    cs[i] = cs_out;
                });
        });

    } else if (
        SolverEOS_LocallyIsothermalLP07 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalLP07>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            auto &mfield = storage.merged_xyzh.get().get(id);

            sham::DeviceBuffer<Tvec> &buf_xyz = mfield.template get_field_buf_ref<Tvec>(0);

            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);

            Tscal cs0  = eos_config->cs0;
            Tscal r0sq = eos_config->r0 * eos_config->r0;
            Tscal mq   = -eos_config->q;

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint, buf_xyz},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [cs0, r0sq, mq](
                    u32 i,
                    auto rho,
                    const Tscal *__restrict U,
                    const Tvec *__restrict xyz,
                    Tscal *__restrict P,
                    Tscal *__restrict cs) {
                    using namespace shamrock::sph;

                    Tvec R      = xyz[i];
                    Tscal rho_a = rho(i);

                    Tscal Rsq    = sycl::dot(R, R);
                    Tscal cs_sq  = EOS::soundspeed_sq(cs0 * cs0, Rsq / r0sq, mq);
                    Tscal cs_out = sycl::sqrt(cs_sq);

                    Tscal P_a = EOS::pressure_from_cs(cs_sq, rho_a);

                    P[i]  = P_a;
                    cs[i] = cs_out;
                });
        });

    } else if (
        SolverEOS_LocallyIsothermalFA2014 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalFA2014>(&solver_config.eos_config.config)) {

        Tscal _G = solver_config.get_constant_G();

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        auto &sink_parts = storage.sinks.get();
        std::vector<Tvec> sink_pos;
        std::vector<Tscal> sink_mass;
        u32 sink_cnt = 0;

        for (auto &s : sink_parts) {
            sink_pos.push_back(s.pos);
            sink_mass.push_back(s.mass);
            sink_cnt++;
        }

        sycl::buffer<Tvec> sink_pos_buf{sink_pos};
        sycl::buffer<Tscal> sink_mass_buf{sink_mass};

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            auto &mfield = storage.merged_xyzh.get().get(id);

            sham::DeviceBuffer<Tvec> &buf_xyz = mfield.template get_field_buf_ref<Tvec>(0);

            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);

            // TODO: Use the complex kernel call when implemented

            sham::EventList depends_list;

            auto P   = buf_P.get_write_access(depends_list);
            auto cs  = buf_cs.get_write_access(depends_list);
            auto rho = rho_getter.get_read_access(depends_list);
            auto U   = buf_uint.get_read_access(depends_list);
            auto xyz = buf_xyz.get_read_access(depends_list);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor spos{sink_pos_buf, cgh, sycl::read_only};
                sycl::accessor smass{sink_mass_buf, cgh, sycl::read_only};
                u32 scount = sink_cnt;

                Tscal h_over_r = eos_config->h_over_r;
                Tscal G        = _G;

                cgh.parallel_for(sycl::range<1>{total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tvec R      = xyz[item];
                    Tscal rho_a = rho(item.get_linear_id());

                    Tscal mpotential = 0;
                    for (u32 i = 0; i < scount; i++) {
                        Tvec s_r      = spos[i] - R;
                        Tscal s_m     = smass[i];
                        Tscal s_r_abs = sycl::length(s_r);
                        mpotential += G * s_m / s_r_abs;
                    }

                    Tscal cs_out = h_over_r * sycl::sqrt(mpotential);
                    Tscal P_a    = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[item]  = P_a;
                    cs[item] = cs_out;
                });
            });

            buf_P.complete_event_state(e);
            buf_cs.complete_event_state(e);
            rho_getter.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_xyz.complete_event_state(e);
        });

    } else if (
        SolverEOS_Tillotson *eos_config
        = std::get_if<SolverEOS_Tillotson>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Tillotson<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P
                = shambase::get_check_ref(storage.pressure).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = shambase::get_check_ref(storage.soundspeed).get_field(id).get_buf();
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);

            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint},
                sham::MultiRef{buf_P, buf_cs},
                total_elements,
                [rho0  = eos_config->rho0,
                 E0    = eos_config->E0,
                 a     = eos_config->a,
                 b     = eos_config->b,
                 A     = eos_config->A,
                 B     = eos_config->B,
                 alpha = eos_config->alpha,
                 beta  = eos_config->beta,
                 u_iv  = eos_config->u_iv,
                 u_cv  = eos_config->u_cv](
                    u32 i,
                    auto rho,
                    const Tscal *__restrict uint,
                    Tscal *__restrict P,
                    Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    auto P_and_cs = EOS::pressure_and_cs(
                        rho(i), uint[i], rho0, A, B, a, b, E0, alpha, beta, u_iv, u_cv);
                    P[i]  = P_and_cs.pressure;
                    cs[i] = P_and_cs.soundspeed;
                });
        });

    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos() {

    NamedStackEntry stack_loc{"compute eos"};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");

    shamrock::SchedulerUtility utility(scheduler());

    shambase::DistributedData<u32> &counts_with_ghosts
        = shambase::get_check_ref(storage.part_counts_with_ghost).indexes;

    shambase::get_check_ref(storage.pressure).ensure_sizes(counts_with_ghosts);
    shambase::get_check_ref(storage.soundspeed).ensure_sizes(counts_with_ghosts);

    if (solver_config.dust_config.has_epsilon_field()) {

        u32 iepsilon_interf = ghost_layout.get_field_idx<Tscal>("epsilon");
        u32 nvar_dust       = solver_config.dust_config.get_dust_nvar();

        compute_eos_internal([&](PatchDataLayer &mpdat) {
            return RhoGetterMonofluid<Tscal>{
                mpdat.get_field_buf_ref<Tscal>(ihpart_interf),
                mpdat.get_field_buf_ref<Tscal>(iepsilon_interf),
                nvar_dust,
                gpart_mass,
                Kernel::hfactd};
        });
    } else {
        compute_eos_internal([&](PatchDataLayer &mpdat) {
            return RhoGetterBase<Tscal>{
                mpdat.get_field_buf_ref<Tscal>(ihpart_interf), gpart_mass, Kernel::hfactd};
        });
    }
}

using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;
template class shammodels::sph::modules::ComputeEos<f64_3, M8>;

template class shammodels::sph::modules::ComputeEos<f64_3, C2>;
template class shammodels::sph::modules::ComputeEos<f64_3, C4>;
template class shammodels::sph::modules::ComputeEos<f64_3, C6>;

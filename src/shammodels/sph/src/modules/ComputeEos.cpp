// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/ComputeEos.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
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

template<class Tscal>
struct RhoGetterSJ {
    sham::DeviceBuffer<Tscal> &buf_h;
    sham::DeviceBuffer<Tscal> &buf_s_j;
    u32 nvar_dust;
    Tscal pmass;
    Tscal hfact;

    struct accessed {
        const Tscal *h;
        const Tscal *buf_s_j;
        u32 nvar_dust;
        Tscal pmass;
        Tscal hfact;

        Tscal operator()(u32 i) const {
            using namespace shamrock::sph;
            Tscal rho = rho_h(pmass, h[i], hfact);

            Tscal epsilon_sum = 0;
            for (u32 j = 0; j < nvar_dust; j++) {
                Tscal s_j = buf_s_j[i * nvar_dust + j];
                epsilon_sum += s_j * s_j / rho;
            }

            return rho * (1 - epsilon_sum);
        }
    };

    accessed get_read_access(sham::EventList &depends_list) {
        auto h   = buf_h.get_read_access(depends_list);
        auto s_j = buf_s_j.get_read_access(depends_list);

        return accessed{h, s_j, nvar_dust, pmass, hfact};
    }

    void complete_event_state(sycl::event e) {
        buf_h.complete_event_state(e);
        buf_s_j.complete_event_state(e);
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos_internal(
    const shamrock::solvergraph::IDataEdge<Tscal> &hfactd,
    const shamrock::solvergraph::IDataEdge<Tscal> &pmass,
    const std::optional<std::reference_wrapper<const shamrock::solvergraph::IFieldSpan<Tscal>>>
        spans_rho,
    const std::optional<std::reference_wrapper<const shamrock::solvergraph::IFieldSpan<Tscal>>>
        spans_h,
    const std::optional<std::reference_wrapper<const shamrock::solvergraph::IFieldSpan<Tscal>>>
        spans_uint,
    const shamrock::solvergraph::Indexes<u32> &sizes,
    shamrock::solvergraph::IFieldSpan<Tscal> &spans_pressure,
    shamrock::solvergraph::IFieldSpan<Tscal> &spans_soundspeed) {

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
    using SolverEOS_LocallyIsothermalFA2014Extended =
        typename SolverConfigEOS::LocallyIsothermalFA2014Extended;
    using SolverEOS_Fermi = typename SolverConfigEOS::Fermi;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    shambase::get_check_ref(storage.pressure).ensure_sizes(sizes.indexes);
    shambase::get_check_ref(storage.soundspeed).ensure_sizes(sizes.indexes);

    bool has_rho = spans_rho.has_value();
    bool has_h   = spans_h.has_value();

    logger::raw_ln("has_rho: ", has_rho, " has_h: ", has_h);

    // must have either rho or h
    if ((!has_rho || has_h) && (has_rho || !has_h)) {
        throw shambase::make_except_with_loc<std::invalid_argument>("Must have either rho or h");
    }

    auto out_refs = sham::DDMultiRef{spans_pressure.get_spans(), spans_soundspeed.get_spans()};

    if (SolverEOS_Isothermal *eos_config
        = std::get_if<SolverEOS_Isothermal>(&solver_config.eos_config.config)) {

        Tscal cs      = eos_config->cs;
        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        using EOS = shamphys::EOS_Isothermal<Tscal>;

        auto eos_internal = [](Tscal cs, Tscal rho, Tscal &pressure, Tscal &soundspeed) {
            using EOS  = shamphys::EOS_Isothermal<Tscal>;
            pressure   = EOS::pressure(cs, rho);
            soundspeed = cs;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans()},
                out_refs,
                sizes.indexes,
                [cs, eos_internal](u32 gid, const Tscal *rho, Tscal *pressure, Tscal *soundspeed) {
                    Tscal rho_a = rho[gid];
                    eos_internal(cs, rho_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans()},
                out_refs,
                sizes.indexes,
                [cs, pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *h, Tscal *pressure, Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tscal rho = rho_h(pmass_, h[gid], hfactd_);
                    eos_internal(cs, rho, pressure[gid], soundspeed[gid]);
                });
        }
    } else if (
        SolverEOS_Adiabatic *eos_config
        = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {

        Tscal gamma   = eos_config->gamma;
        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        using EOS = shamphys::EOS_Adiabatic<Tscal>;

        auto eos_internal
            = [](Tscal gamma, Tscal rho, Tscal uint, Tscal &pressure, Tscal &soundspeed) {
                  using EOS  = shamphys::EOS_Adiabatic<Tscal>;
                  Tscal P_a  = EOS::pressure(gamma, rho, uint);
                  Tscal cs_a = EOS::cs_from_p(gamma, rho, P_a);
                  pressure   = P_a;
                  soundspeed = cs_a;
              };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans(), spans_uint.value().get().get_spans()},
                out_refs,
                sizes.indexes,
                [gamma, pmass_, hfactd_, eos_internal](
                    u32 gid,
                    const Tscal *rho,
                    const Tscal *uint,
                    Tscal *pressure,
                    Tscal *soundspeed) {
                    Tscal rho_a  = rho[gid];
                    Tscal uint_a = uint[gid];
                    eos_internal(gamma, rho_a, uint_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans(), spans_uint.value().get().get_spans()},
                out_refs,
                sizes.indexes,
                [gamma, pmass_, hfactd_, eos_internal](
                    u32 gid,
                    const Tscal *h,
                    const Tscal *uint,
                    Tscal *pressure,
                    Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tscal rho    = rho_h(pmass_, h[gid], hfactd_);
                    Tscal uint_a = uint[gid];
                    eos_internal(gamma, rho, uint_a, pressure[gid], soundspeed[gid]);
                });
        }

    } else if (
        SolverEOS_Polytropic *eos_config
        = std::get_if<SolverEOS_Polytropic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Polytropic<Tscal>;

        Tscal K       = eos_config->K;
        Tscal gamma   = eos_config->gamma;
        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        auto eos_internal
            = [](Tscal K, Tscal gamma, Tscal rho_a, Tscal &pressure, Tscal &soundspeed) {
                  using EOS  = shamphys::EOS_Polytropic<Tscal>;
                  Tscal P_a  = EOS::pressure(gamma, K, rho_a);
                  Tscal cs_a = EOS::soundspeed(gamma, K, rho_a);
                  pressure   = P_a;
                  soundspeed = cs_a;
              };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans()},
                out_refs,
                sizes.indexes,
                [K, gamma, pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *rho, Tscal *pressure, Tscal *soundspeed) {
                    Tscal rho_a = rho[gid];
                    eos_internal(K, gamma, rho_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans()},
                out_refs,
                sizes.indexes,
                [K, gamma, pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *h, Tscal *pressure, Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tscal rho = rho_h(pmass_, h[gid], hfactd_);
                    eos_internal(K, gamma, rho, pressure[gid], soundspeed[gid]);
                });
        }

    } else if (
        SolverEOS_LocallyIsothermal *eos_config
        = std::get_if<SolverEOS_LocallyIsothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        u32 isoundspeed_interf = ghost_layout.get_field_idx<Tscal>("soundspeed");

        shamrock::solvergraph::FieldRefs<Tscal> soundspeed_refs{"", ""};
        auto refs = storage.merged_patchdata_ghost.get()
                        .template map<shamrock::solvergraph::PatchDataFieldRef<Tscal>>(
                            [&](u64 id, PatchDataLayer &mpdat)
                                -> shamrock::solvergraph::PatchDataFieldRef<Tscal> {
                                return mpdat.get_field<Tscal>(isoundspeed_interf);
                            });
        soundspeed_refs.set_refs(refs);

        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        auto eos_internal = [](Tscal cs0, Tscal rho_a, Tscal &pressure, Tscal &soundspeed) {
            using EOS  = shamphys::EOS_LocallyIsothermal<Tscal>;
            pressure   = EOS::pressure_from_cs(cs0 * cs0, rho_a);
            soundspeed = cs0;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans(), soundspeed_refs.get_spans()},
                out_refs,
                sizes.indexes,
                [eos_internal](
                    u32 gid,
                    const Tscal *rho,
                    const Tscal *cs0,
                    Tscal *pressure,
                    Tscal *soundspeed) {
                    Tscal rho_a = rho[gid];
                    Tscal cs0_a = cs0[gid];
                    eos_internal(cs0_a, rho_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans(), soundspeed_refs.get_spans()},
                out_refs,
                sizes.indexes,
                [pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *h, const Tscal *cs0, Tscal *pressure, Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tscal rho   = rho_h(pmass_, h[gid], hfactd_);
                    Tscal cs0_a = cs0[gid];
                    eos_internal(cs0_a, rho, pressure[gid], soundspeed[gid]);
                });
        }

    } else if (
        SolverEOS_LocallyIsothermalLP07 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalLP07>(&solver_config.eos_config.config)) {

        Tscal cs0  = eos_config->cs0;
        Tscal r0sq = eos_config->r0 * eos_config->r0;
        Tscal mq   = -eos_config->q;

        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        shamrock::solvergraph::FieldRefs<Tvec> xyz_refs{"", ""};
        auto refs
            = storage.merged_patchdata_ghost.get()
                  .template map<shamrock::solvergraph::PatchDataFieldRef<Tvec>>(
                      [&](u64 id,
                          PatchDataLayer &mpdat) -> shamrock::solvergraph::PatchDataFieldRef<Tvec> {
                          return mpdat.get_field<Tvec>(0);
                      });
        xyz_refs.set_refs(refs);

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        auto eos_internal = [](Tvec R,
                               Tscal cs0,
                               Tscal r0sq,
                               Tscal mq,
                               Tscal rho_a,
                               Tscal &pressure,
                               Tscal &soundspeed) {
            Tscal Rsq    = sycl::dot(R, R);
            Tscal cs_sq  = EOS::soundspeed_sq(cs0 * cs0, Rsq / r0sq, mq);
            Tscal cs_out = sycl::sqrt(cs_sq);

            Tscal P_a = EOS::pressure_from_cs(cs_sq, rho_a);

            pressure   = P_a;
            soundspeed = cs_out;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans(), xyz_refs.get_spans()},
                out_refs,
                sizes.indexes,
                [cs0, r0sq, mq, eos_internal](
                    u32 gid,
                    const Tscal *rho,
                    const Tvec *xyz,
                    Tscal *pressure,
                    Tscal *soundspeed) {
                    Tvec R_a    = xyz[gid];
                    Tscal rho_a = rho[gid];
                    eos_internal(R_a, cs0, r0sq, mq, rho_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans(), xyz_refs.get_spans()},
                out_refs,
                sizes.indexes,
                [cs0, r0sq, mq, pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *h, const Tvec *xyz, Tscal *pressure, Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tvec R_a    = xyz[gid];
                    Tscal rho_a = rho_h(pmass_, h[gid], hfactd_);
                    eos_internal(R_a, cs0, r0sq, mq, rho_a, pressure[gid], soundspeed[gid]);
                });
        }

    } else if (
        SolverEOS_LocallyIsothermalFA2014 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalFA2014>(&solver_config.eos_config.config)) {

        Tscal G        = solver_config.get_constant_G();
        Tscal h_over_r = eos_config->h_over_r;
        Tscal pmass_   = pmass.data;
        Tscal hfactd_  = hfactd.data;

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

        shamrock::solvergraph::FieldRefs<Tvec> xyz_refs{"", ""};
        auto refs
            = storage.merged_patchdata_ghost.get()
                  .template map<shamrock::solvergraph::PatchDataFieldRef<Tvec>>(
                      [&](u64 id,
                          PatchDataLayer &mpdat) -> shamrock::solvergraph::PatchDataFieldRef<Tvec> {
                          return mpdat.get_field<Tvec>(0);
                      });
        xyz_refs.set_refs(refs);

        sham::DeviceBuffer<Tvec> sink_pos_buf(sink_pos.size(), dev_sched);
        sham::DeviceBuffer<Tscal> sink_mass_buf(sink_mass.size(), dev_sched);

        sink_pos_buf.copy_from_stdvec(sink_pos);
        sink_mass_buf.copy_from_stdvec(sink_mass);

        auto eos_internal = [](Tvec R,
                               Tscal rho_a,
                               u32 scount,
                               auto spos,
                               auto smass,
                               Tscal G,
                               Tscal h_over_r,
                               Tscal &pressure,
                               Tscal &soundspeed) {
            Tscal mpotential = 0;
            for (u32 i = 0; i < scount; i++) {
                Tvec s_r      = spos[i] - R;
                Tscal s_m     = smass[i];
                Tscal s_r_abs = sycl::length(s_r);
                mpotential += G * s_m / s_r_abs;
            }

            Tscal cs_out = h_over_r * sycl::sqrt(mpotential);
            Tscal P_a    = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

            pressure   = P_a;
            soundspeed = cs_out;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);

            sizes.indexes.for_each([&](u64 id, u32 count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        spans_rho_.get_spans().get(id),
                        xyz_refs.get_spans().get(id),
                        sink_pos_buf,
                        sink_mass_buf},
                    out_refs.get(id),
                    count,
                    [G, h_over_r, sink_cnt, eos_internal](
                        u32 gid,
                        const Tscal *rho,
                        const Tvec *xyz,
                        const Tvec *spos,
                        const Tscal *smass,
                        Tscal *pressure,
                        Tscal *soundspeed) {
                        Tvec R_a    = xyz[gid];
                        Tscal rho_a = rho[gid];
                        eos_internal(
                            R_a,
                            rho_a,
                            sink_cnt,
                            spos,
                            smass,
                            G,
                            h_over_r,
                            pressure[gid],
                            soundspeed[gid]);
                    });
            });

        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);

            sizes.indexes.for_each([&](u64 id, u32 count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        spans_h_.get_spans().get(id),
                        xyz_refs.get_spans().get(id),
                        sink_pos_buf,
                        sink_mass_buf},
                    out_refs.get(id),
                    count,
                    [G, h_over_r, sink_cnt, pmass_, hfactd_, eos_internal](
                        u32 gid,
                        const Tscal *h,
                        const Tvec *xyz,
                        const Tvec *spos,
                        const Tscal *smass,
                        Tscal *pressure,
                        Tscal *soundspeed) {
                        using namespace shamrock::sph;
                        Tvec R_a    = xyz[gid];
                        Tscal rho_a = rho_h(pmass_, h[gid], hfactd_);
                        eos_internal(
                            R_a,
                            rho_a,
                            sink_cnt,
                            spos,
                            smass,
                            G,
                            h_over_r,
                            pressure[gid],
                            soundspeed[gid]);
                    });
            });
        }

    } else if (
        SolverEOS_LocallyIsothermalFA2014Extended *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalFA2014Extended>(
            &solver_config.eos_config.config)) {

        Tscal cs0     = eos_config->cs0;
        Tscal r0      = eos_config->r0;
        Tscal q_      = eos_config->q;
        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;
        u32 n_sinks   = eos_config->n_sinks;

        Tscal inv_r0_q = 1. / sycl::pow(r0, q_);

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        auto &sink_parts = storage.sinks.get();
        std::vector<Tvec> sink_pos;
        std::vector<Tscal> sink_mass;
        u32 sink_cnt = 0;

        for (auto &s : sink_parts) {
            sink_pos.push_back(s.pos);
            sink_mass.push_back(s.mass);
            sink_cnt++;
            if (sink_pos.size() >= n_sinks) { // We only consider the first n_sinks sinks
                break;
            }
        }

        if (sink_cnt == 0) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "No sinks found for the equation of state");
        }

        shamrock::solvergraph::FieldRefs<Tvec> xyz_refs{"", ""};
        auto refs
            = storage.merged_patchdata_ghost.get()
                  .template map<shamrock::solvergraph::PatchDataFieldRef<Tvec>>(
                      [&](u64 id,
                          PatchDataLayer &mpdat) -> shamrock::solvergraph::PatchDataFieldRef<Tvec> {
                          return mpdat.get_field<Tvec>(0);
                      });
        xyz_refs.set_refs(refs);

        sham::DeviceBuffer<Tvec> sink_pos_buf(sink_pos.size(), dev_sched);
        sham::DeviceBuffer<Tscal> sink_mass_buf(sink_mass.size(), dev_sched);

        sink_pos_buf.copy_from_stdvec(sink_pos);
        sink_mass_buf.copy_from_stdvec(sink_mass);

        auto eos_internal = [](Tvec R,
                               Tscal rho_a,
                               u32 scount,
                               auto spos,
                               auto smass,
                               Tscal cs0,
                               Tscal inv_r0_q,
                               Tscal q,
                               Tscal &pressure,
                               Tscal &soundspeed) {
            Tscal sink_mass_sum = 0;
            Tscal pot_sum       = 0;
            for (u32 i = 0; i < scount; i++) {
                Tvec s_r      = spos[i] - R;
                Tscal s_m     = smass[i];
                Tscal s_r_abs = sycl::length(s_r);
                sink_mass_sum += s_m;
                pot_sum += s_m / s_r_abs;
            }

            Tscal cs_out = cs0 * inv_r0_q * sycl::pow(pot_sum / sink_mass_sum, q);
            Tscal P_a    = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

            pressure   = P_a;
            soundspeed = cs_out;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);

            sizes.indexes.for_each([&](u64 id, u32 count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        spans_rho_.get_spans().get(id),
                        xyz_refs.get_spans().get(id),
                        sink_pos_buf,
                        sink_mass_buf},
                    out_refs.get(id),
                    count,
                    [cs0, inv_r0_q, q_, sink_cnt, eos_internal](
                        u32 gid,
                        const Tscal *rho,
                        const Tvec *xyz,
                        const Tvec *spos,
                        const Tscal *smass,
                        Tscal *pressure,
                        Tscal *soundspeed) {
                        Tvec R_a    = xyz[gid];
                        Tscal rho_a = rho[gid];
                        eos_internal(
                            R_a,
                            rho_a,
                            sink_cnt,
                            spos,
                            smass,
                            cs0,
                            inv_r0_q,
                            q_,
                            pressure[gid],
                            soundspeed[gid]);
                    });
            });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);

            sizes.indexes.for_each([&](u64 id, u32 count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        spans_h_.get_spans().get(id),
                        xyz_refs.get_spans().get(id),
                        sink_pos_buf,
                        sink_mass_buf},
                    out_refs.get(id),
                    count,
                    [cs0, inv_r0_q, q_, sink_cnt, pmass_, hfactd_, eos_internal](
                        u32 gid,
                        const Tscal *h,
                        const Tvec *xyz,
                        const Tvec *spos,
                        const Tscal *smass,
                        Tscal *pressure,
                        Tscal *soundspeed) {
                        using namespace shamrock::sph;
                        Tvec R_a    = xyz[gid];
                        Tscal rho_a = rho_h(pmass_, h[gid], hfactd_);
                        eos_internal(
                            R_a,
                            rho_a,
                            sink_cnt,
                            spos,
                            smass,
                            cs0,
                            inv_r0_q,
                            q_,
                            pressure[gid],
                            soundspeed[gid]);
                    });
            });
        }

    } else if (
        SolverEOS_Fermi *eos_config
        = std::get_if<SolverEOS_Fermi>(&solver_config.eos_config.config)) {

        using namespace shamunits;
        auto unit_sys = *solver_config.unit_sys;

        Tscal mass   = unit_sys.template to<units::kilogram>();
        Tscal length = unit_sys.template to<units::metre>();
        Tscal time   = unit_sys.template to<units::second>();

        Tscal pressure_unit = mass / length / (time * time);
        Tscal density_unit  = mass / (length * length * length);
        Tscal velocity_unit = length / time;

        Tscal mu_e = eos_config->mu_e;

        Tscal pmass_  = pmass.data;
        Tscal hfactd_ = hfactd.data;

        auto eos_internal = [density_unit, pressure_unit, velocity_unit](
                                Tscal mu_e, Tscal rho_a, Tscal &pressure, Tscal &soundspeed) {
            using EOS      = shamphys::EOS_Fermi<Tscal>;
            auto const res = EOS::pressure_and_soundspeed(mu_e, rho_a * density_unit);
            pressure       = res.pressure / pressure_unit;
            soundspeed     = res.soundspeed / velocity_unit;
        };

        if (has_rho) {
            auto &spans_rho_ = spans_rho.value().get();
            spans_rho_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_rho_.get_spans()},
                out_refs,
                sizes.indexes,
                [mu_e,
                 eos_internal](u32 gid, const Tscal *rho, Tscal *pressure, Tscal *soundspeed) {
                    Tscal rho_a = rho[gid];
                    eos_internal(mu_e, rho_a, pressure[gid], soundspeed[gid]);
                });
        } else if (has_h) {
            auto &spans_h_ = spans_h.value().get();
            spans_h_.check_sizes(sizes.indexes);
            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{spans_h_.get_spans()},
                out_refs,
                sizes.indexes,
                [mu_e, pmass_, hfactd_, eos_internal](
                    u32 gid, const Tscal *h, Tscal *pressure, Tscal *soundspeed) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho_h(pmass_, h[gid], hfactd_);
                    eos_internal(mu_e, rho_a, pressure[gid], soundspeed[gid]);
                });
        }

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
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::solvergraph::IDataEdge<Tscal> hfactd("hfactd", "hfactd");
    shamrock::solvergraph::IDataEdge<Tscal> pmass("pmass", "pmass");

    hfactd.data = Kernel::hfactd;
    pmass.data  = gpart_mass;

    auto &sizes = shambase::get_check_ref(storage.part_counts_with_ghost);

    shamrock::solvergraph::FieldRefs<Tscal> h_refs{"", ""};
    {
        auto refs = storage.merged_patchdata_ghost.get()
                        .template map<shamrock::solvergraph::PatchDataFieldRef<Tscal>>(
                            [&](u64 id, PatchDataLayer &mpdat)
                                -> shamrock::solvergraph::PatchDataFieldRef<Tscal> {
                                return mpdat.get_field<Tscal>(ihpart_interf);
                            });
        h_refs.set_refs(refs);
    }

    shamrock::solvergraph::FieldRefs<Tscal> uint_refs{"", ""};
    {
        auto refs = storage.merged_patchdata_ghost.get()
                        .template map<shamrock::solvergraph::PatchDataFieldRef<Tscal>>(
                            [&](u64 id, PatchDataLayer &mpdat)
                                -> shamrock::solvergraph::PatchDataFieldRef<Tscal> {
                                return mpdat.get_field<Tscal>(iuint_interf);
                            });
        uint_refs.set_refs(refs);
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    if (solver_config.dust_config.has_epsilon_field()) {

        u32 iepsilon_interf = ghost_layout.get_field_idx<Tscal>("epsilon");
        u32 nvar_dust       = solver_config.dust_config.get_dust_nvar();

        shamrock::solvergraph::Field<Tscal> rho_g
            = shamrock::solvergraph::Field<Tscal>(1, "rho_g", "rho_g");
        shamrock::solvergraph::Field<Tscal> uint_g
            = shamrock::solvergraph::Field<Tscal>(nvar_dust, "uint_g", "uint_g");

        rho_g.ensure_sizes(sizes.indexes);
        uint_g.ensure_sizes(sizes.indexes);

        shamrock::solvergraph::FieldRefs<Tscal> epsilon_refs{"", ""};
        auto refs = storage.merged_patchdata_ghost.get()
                        .template map<shamrock::solvergraph::PatchDataFieldRef<Tscal>>(
                            [&](u64 id, PatchDataLayer &mpdat)
                                -> shamrock::solvergraph::PatchDataFieldRef<Tscal> {
                                return mpdat.get_field<Tscal>(iepsilon_interf);
                            });
        epsilon_refs.set_refs(refs);

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{h_refs.get_spans(), uint_refs.get_spans(), epsilon_refs.get_spans()},
            sham::DDMultiRef{rho_g.get_spans(), uint_g.get_spans()},
            sizes.indexes,
            [pmass = pmass.data, hfactd = hfactd.data, nvar_dust](
                u32 gid,
                const Tscal *h,
                const Tscal *uint,
                const Tscal *epsilon,
                Tscal *rho_g,
                Tscal *uint_g) {
                using namespace shamrock::sph;
                Tscal rho_a  = rho_h(pmass, h[gid], hfactd);
                Tscal uint_a = uint[gid];

                Tscal epsilon_sum = 0;
                for (u32 j = 0; j < nvar_dust; j++) {
                    epsilon_sum += epsilon[gid * nvar_dust + j];
                }

                Tscal rho_g_a  = rho_a * (1 - epsilon_sum);
                Tscal uint_g_a = uint_a / (1 - epsilon_sum);

                rho_g[gid]  = rho_g_a;
                uint_g[gid] = uint_g_a;
            });

        compute_eos_internal(
            hfactd,
            pmass,
            rho_g,
            std::nullopt,
            uint_g,
            sizes,
            shambase::get_check_ref(storage.pressure),
            shambase::get_check_ref(storage.soundspeed));
    } else if (solver_config.dust_config.has_s_j_field()) {

        u32 is_j_interf = ghost_layout.get_field_idx<Tscal>("s_j");
        u32 nvar_dust   = solver_config.dust_config.get_dust_nvar();

        shamrock::solvergraph::Field<Tscal> rho_g
            = shamrock::solvergraph::Field<Tscal>(1, "rho_g", "rho_g");
        shamrock::solvergraph::Field<Tscal> uint_g
            = shamrock::solvergraph::Field<Tscal>(nvar_dust, "uint_g", "uint_g");

        rho_g.ensure_sizes(sizes.indexes);
        uint_g.ensure_sizes(sizes.indexes);

        shamrock::solvergraph::FieldRefs<Tscal> s_j_refs{"", ""};
        auto refs = storage.merged_patchdata_ghost.get()
                        .template map<shamrock::solvergraph::PatchDataFieldRef<Tscal>>(
                            [&](u64 id, PatchDataLayer &mpdat)
                                -> shamrock::solvergraph::PatchDataFieldRef<Tscal> {
                                return mpdat.get_field<Tscal>(is_j_interf);
                            });
        s_j_refs.set_refs(refs);

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{h_refs.get_spans(), uint_refs.get_spans(), s_j_refs.get_spans()},
            sham::DDMultiRef{rho_g.get_spans(), uint_g.get_spans()},
            sizes.indexes,
            [pmass = pmass.data, hfactd = hfactd.data, nvar_dust](
                u32 gid,
                const Tscal *h,
                const Tscal *uint,
                const Tscal *s_j,
                Tscal *rho_g,
                Tscal *uint_g) {
                using namespace shamrock::sph;
                Tscal rho_a  = rho_h(pmass, h[gid], hfactd);
                Tscal uint_a = uint[gid];

                Tscal epsilon_sum = 0;
                for (u32 j = 0; j < nvar_dust; j++) {
                    Tscal s = s_j[gid * nvar_dust + j];
                    epsilon_sum += s * s / rho_a;
                }

                Tscal rho_g_a  = rho_a * (1 - epsilon_sum);
                Tscal uint_g_a = uint_a / (1 - epsilon_sum);

                rho_g[gid]  = rho_g_a;
                uint_g[gid] = uint_g_a;
            });

        compute_eos_internal(
            hfactd,
            pmass,
            rho_g,
            std::nullopt,
            uint_g,
            sizes,
            shambase::get_check_ref(storage.pressure),
            shambase::get_check_ref(storage.soundspeed));
    } else {

        compute_eos_internal(
            hfactd,
            pmass,
            std::nullopt,
            h_refs,
            uint_refs,
            sizes,
            shambase::get_check_ref(storage.pressure),
            shambase::get_check_ref(storage.soundspeed));
    }
}

using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;
template class shammodels::sph::modules::ComputeEos<f64_3, M8>;

template class shammodels::sph::modules::ComputeEos<f64_3, C2>;
template class shammodels::sph::modules::ComputeEos<f64_3, C4>;
template class shammodels::sph::modules::ComputeEos<f64_3, C6>;

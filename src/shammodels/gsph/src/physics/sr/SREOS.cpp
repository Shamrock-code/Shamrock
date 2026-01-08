// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SREOS.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief EOS computation implementation for SR-GSPH
 */

#include "shambase/stacktrace.hpp"
#include "shammodels/gsph/physics/sr/SREOS.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::physics::sr {

    template<class Tvec, template<class> class SPHKernel>
    void SREOS<Tvec, SPHKernel>::compute(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        auto dev_sched      = shamsys::instance::get_compute_scheduler_ptr();
        const Tscal gamma   = config.get_eos_gamma();
        const bool has_uint = config.has_field_uint();
        const Tscal c_speed = config.c_speed;
        const Tscal c2      = c_speed * c_speed;

        PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(storage.ghost_layout.get());
        u32 idensity_interf                = ghost_layout.get_field_idx<Tscal>("density");
        u32 iuint_interf = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;
        u32 ivxyz_interf = ghost_layout.get_field_idx<Tvec>("vxyz");

        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);
        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage.soundspeed);

        shambase::DistributedData<u32> &counts_with_ghosts
            = shambase::get_check_ref(storage.part_counts_with_ghost).indexes;

        pressure_field.ensure_sizes(counts_with_ghosts);
        soundspeed_field.ensure_sizes(counts_with_ghosts);

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);
            if (total_elements == 0)
                return;

            sham::DeviceBuffer<Tscal> &buf_density
                = mpdat.get_field_buf_ref<Tscal>(idensity_interf);
            sham::DeviceBuffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            auto &pressure_buf                 = pressure_field.get_field(id).get_buf();
            auto &soundspeed_buf               = soundspeed_field.get_field(id).get_buf();

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto density    = buf_density.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto pressure   = pressure_buf.get_write_access(depends_list);
            auto soundspeed = soundspeed_buf.get_write_access(depends_list);

            const Tscal *uint_ptr = nullptr;
            if (has_uint) {
                uint_ptr
                    = mpdat.get_field_buf_ref<Tscal>(iuint_interf).get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, total_elements, "sr_compute_eos", [=](u64 gid) {
                    u32 i = (u32) gid;

                    // density field contains lab-frame N from compute_omega_sr()
                    // We need rest-frame n = N / gamma
                    Tscal N_lab = density[i];

                    // Compute Lorentz factor from velocity
                    Tvec v_i         = vxyz[i];
                    Tscal v2         = sycl::dot(v_i, v_i) / c2;
                    Tscal gamma_lor  = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2, Tscal{1e-10}));

                    // Rest-frame density: n = N / gamma
                    Tscal n = N_lab / gamma_lor;

                    if (has_uint && uint_ptr != nullptr) {
                        Tscal u = uint_ptr[i];

                        // P = (gamma_eos - 1) * n * u for adiabatic EOS
                        Tscal P = (gamma - Tscal{1}) * n * u;

                        const Tscal H   = Tscal{1} + u / c2 + P / (n * c2);
                        const Tscal cs2 = (gamma - Tscal{1}) * (H - Tscal{1}) / H;
                        Tscal cs        = sycl::sqrt(sycl::fmax(cs2, Tscal{0})) * c_speed;

                        pressure[i]   = P;
                        soundspeed[i] = cs;
                    } else {
                        Tscal cs = c_speed * Tscal{0.1};
                        Tscal P  = cs * cs * n;

                        pressure[i]   = P;
                        soundspeed[i] = cs;
                    }
                });
            });

            buf_density.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            if (has_uint) {
                mpdat.get_field_buf_ref<Tscal>(iuint_interf).complete_event_state(e);
            }
            pressure_buf.complete_event_state(e);
            soundspeed_buf.complete_event_state(e);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SREOS<Tvec, SPHKernel>::compute_output_density_field(
        PatchScheduler &scheduler,
        const Config &config,
        Tscal c_speed,
        shamrock::ComputeField<Tscal> &density) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        const bool has_pmass  = config.has_field_pmass();
        const u32 ipmass      = has_pmass ? pdl.get_field_idx<Tscal>("pmass") : 0;
        const Tscal part_mass = config.gpart_mass;
        const Tscal c         = c_speed;

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();
            auto &buf_vxyz  = pdat.get_field<Tvec>(ivxyz).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
            const Tvec *acc_v  = buf_vxyz.get_read_access(depends_list);
            auto acc_rho       = density.get_buf(p.id_patch).get_write_access(depends_list);

            const Tscal *acc_pmass                   = nullptr;
            sham::DeviceBuffer<Tscal> *buf_pmass_ptr = nullptr;
            if (has_pmass) {
                buf_pmass_ptr = &pdat.get_field<Tscal>(ipmass).get_buf();
                acc_pmass     = buf_pmass_ptr->get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid = (u32) item.get_id();
                    using namespace shamrock::sph;

                    Tscal m = has_pmass ? acc_pmass[gid] : part_mass;
                    Tscal N = rho_h(m, acc_h[gid], Kernel::hfactd);

                    Tvec v       = acc_v[gid];
                    Tscal v2     = sycl::dot(v, v) / (c * c);
                    Tscal gamma  = Tscal{1} / sycl::sqrt(Tscal{1} - v2);
                    acc_rho[gid] = N / gamma;
                });
            });

            buf_hpart.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            density.get_buf(p.id_patch).complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SREOS<Tvec, SPHKernel>::compute_output_pressure_field(
        PatchScheduler &scheduler,
        const Config &config,
        Tscal c_speed,
        shamrock::ComputeField<Tscal> &pressure) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        const bool has_uint  = config.has_field_uint();
        const u32 iuint      = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
        const bool has_pmass = config.has_field_pmass();
        const u32 ipmass     = has_pmass ? pdl.get_field_idx<Tscal>("pmass") : 0;

        const Tscal part_mass = config.gpart_mass;
        const Tscal gamma_eos = config.get_eos_gamma();
        const Tscal c         = c_speed;

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();
            auto &buf_vxyz  = pdat.get_field<Tvec>(ivxyz).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
            const Tvec *acc_v  = buf_vxyz.get_read_access(depends_list);
            auto acc_P         = pressure.get_buf(p.id_patch).get_write_access(depends_list);

            const Tscal *acc_u                      = nullptr;
            sham::DeviceBuffer<Tscal> *buf_uint_ptr = nullptr;
            if (has_uint) {
                buf_uint_ptr = &pdat.get_field<Tscal>(iuint).get_buf();
                acc_u        = buf_uint_ptr->get_read_access(depends_list);
            }

            const Tscal *acc_pmass                   = nullptr;
            sham::DeviceBuffer<Tscal> *buf_pmass_ptr = nullptr;
            if (has_pmass) {
                buf_pmass_ptr = &pdat.get_field<Tscal>(ipmass).get_buf();
                acc_pmass     = buf_pmass_ptr->get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid = (u32) item.get_id();
                    using namespace shamrock::sph;

                    Tscal m = has_pmass ? acc_pmass[gid] : part_mass;
                    Tscal N = rho_h(m, acc_h[gid], Kernel::hfactd);

                    Tvec v       = acc_v[gid];
                    Tscal v2     = sycl::dot(v, v) / (c * c);
                    Tscal gamma  = Tscal{1} / sycl::sqrt(Tscal{1} - v2);
                    Tscal n_rest = N / gamma;

                    if (has_uint && acc_u != nullptr) {
                        acc_P[gid] = (gamma_eos - Tscal{1}) * n_rest * acc_u[gid];
                    } else {
                        Tscal cs   = Tscal{0.1};
                        acc_P[gid] = cs * cs * n_rest;
                    }
                });
            });

            buf_hpart.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            pressure.get_buf(p.id_patch).complete_event_state(e);
            if (has_uint && buf_uint_ptr) {
                buf_uint_ptr->complete_event_state(e);
            }
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });
    }

    // Explicit instantiations
    using namespace shammath;

    template class SREOS<sycl::vec<double, 3>, M4>;
    template class SREOS<sycl::vec<double, 3>, M6>;
    template class SREOS<sycl::vec<double, 3>, M8>;
    template class SREOS<sycl::vec<double, 3>, C2>;
    template class SREOS<sycl::vec<double, 3>, C4>;
    template class SREOS<sycl::vec<double, 3>, C6>;
    template class SREOS<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::sr

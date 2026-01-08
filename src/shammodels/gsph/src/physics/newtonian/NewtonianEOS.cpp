// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NewtonianEOS.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief EOS computation implementation for Newtonian GSPH
 */

#include "shambase/stacktrace.hpp"
#include "shammodels/gsph/FieldNames.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianEOS.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::physics::newtonian {

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianEOS<Tvec, SPHKernel>::compute(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;
        using namespace shammodels::gsph;

        auto dev_sched      = shamsys::instance::get_compute_scheduler_ptr();
        const Tscal gamma   = config.get_eos_gamma();
        const bool has_uint = config.has_field_uint();

        PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(storage.ghost_layout.get());
        u32 idensity_interf                = ghost_layout.get_field_idx<Tscal>(computed_fields::DENSITY);
        u32 iuint_interf = has_uint ? ghost_layout.get_field_idx<Tscal>(fields::UINT) : 0;

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
            auto &pressure_buf   = pressure_field.get_field(id).get_buf();
            auto &soundspeed_buf = soundspeed_field.get_field(id).get_buf();

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto density = buf_density.get_read_access(depends_list);
            auto pressure   = pressure_buf.get_write_access(depends_list);
            auto soundspeed = soundspeed_buf.get_write_access(depends_list);

            const Tscal *uint_ptr = nullptr;
            if (has_uint) {
                uint_ptr
                    = mpdat.get_field_buf_ref<Tscal>(iuint_interf).get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, total_elements, "compute_eos_newtonian", [=](u64 gid) {
                    u32 i     = (u32) gid;
                    Tscal rho = sycl::fmax(density[i], Tscal{1e-30});

                    if (has_uint && uint_ptr != nullptr) {
                        Tscal u  = sycl::fmax(uint_ptr[i], Tscal{1e-30});
                        Tscal P  = (gamma - Tscal{1}) * rho * u;
                        Tscal cs = sycl::sqrt(gamma * (gamma - Tscal{1}) * u);

                        pressure[i]   = sycl::clamp(P, Tscal{1e-30}, Tscal{1e30});
                        soundspeed[i] = sycl::clamp(cs, Tscal{1e-10}, Tscal{1e10});
                    } else {
                        // Isothermal EOS
                        Tscal cs      = Tscal{1.0};
                        pressure[i]   = cs * cs * rho;
                        soundspeed[i] = cs;
                    }
                });
            });

            buf_density.complete_event_state(e);
            if (has_uint) {
                mpdat.get_field_buf_ref<Tscal>(iuint_interf).complete_event_state(e);
            }
            pressure_buf.complete_event_state(e);
            soundspeed_buf.complete_event_state(e);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianEOS<Tvec, SPHKernel>::compute_output_density_field(
        PatchScheduler &scheduler, const Config &config, shamrock::ComputeField<Tscal> &density) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>(fields::HPART);

        const bool has_pmass  = config.has_field_pmass();
        const u32 ipmass      = has_pmass ? pdl.get_field_idx<Tscal>("pmass") : 0;
        const Tscal part_mass = config.gpart_mass;

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
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

                    Tscal m      = has_pmass ? acc_pmass[gid] : part_mass;
                    acc_rho[gid] = rho_h(m, acc_h[gid], Kernel::hfactd);
                });
            });

            buf_hpart.complete_event_state(e);
            density.get_buf(p.id_patch).complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianEOS<Tvec, SPHKernel>::compute_output_pressure_field(
        PatchScheduler &scheduler, const Config &config, shamrock::ComputeField<Tscal> &pressure) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");

        const bool has_uint  = config.has_field_uint();
        const u32 iuint      = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
        const bool has_pmass = config.has_field_pmass();
        const u32 ipmass     = has_pmass ? pdl.get_field_idx<Tscal>("pmass") : 0;

        const Tscal part_mass = config.gpart_mass;
        const Tscal gamma_eos = config.get_eos_gamma();

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
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

                    Tscal m   = has_pmass ? acc_pmass[gid] : part_mass;
                    Tscal rho = rho_h(m, acc_h[gid], Kernel::hfactd);

                    if (has_uint && acc_u != nullptr) {
                        acc_P[gid] = (gamma_eos - Tscal{1}) * rho * acc_u[gid];
                    } else {
                        Tscal cs   = Tscal{0.1};
                        acc_P[gid] = cs * cs * rho;
                    }
                });
            });

            buf_hpart.complete_event_state(e);
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

    template class NewtonianEOS<sycl::vec<double, 3>, M4>;
    template class NewtonianEOS<sycl::vec<double, 3>, M6>;
    template class NewtonianEOS<sycl::vec<double, 3>, M8>;
    template class NewtonianEOS<sycl::vec<double, 3>, C2>;
    template class NewtonianEOS<sycl::vec<double, 3>, C4>;
    template class NewtonianEOS<sycl::vec<double, 3>, C6>;
    template class NewtonianEOS<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::newtonian

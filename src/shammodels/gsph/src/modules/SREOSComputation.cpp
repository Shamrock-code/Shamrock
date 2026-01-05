// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SREOSComputation.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Implementation of SR-GSPH EOS computation module
 */

#include "shammodels/gsph/modules/SREOSComputation.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambase/stacktrace.hpp"
#include "shammath/sphkernels.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void SREOSComputation<Tvec, SPHKernel>::compute_eos() {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;

        auto dev_sched      = shamsys::instance::get_compute_scheduler_ptr();
        const Tscal gamma   = solver_config.get_eos_gamma();
        const bool has_uint = solver_config.has_field_uint();
        const Tscal c_speed = solver_config.sr_config.get_c_speed();
        const Tscal c2      = c_speed * c_speed;

        // Get ghost layout field indices
        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage.ghost_layout.get());
        u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
        u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;
        u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");

        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);
        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage.soundspeed);

        // Size buffers to part_counts_with_ghost (includes ghosts!)
        shambase::DistributedData<u32> &counts_with_ghosts
            = shambase::get_check_ref(storage.part_counts_with_ghost).indexes;

        pressure_field.ensure_sizes(counts_with_ghosts);
        soundspeed_field.ensure_sizes(counts_with_ghosts);

        // Iterate over merged_patchdata_ghost (includes local + ghost particles)
        storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
            u32 total_elements
                = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);
            if (total_elements == 0)
                return;

            // Get buffers
            sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);
            sham::DeviceBuffer<Tvec> &buf_vxyz     = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            auto &pressure_buf                     = pressure_field.get_field(id).get_buf();
            auto &soundspeed_buf                   = soundspeed_field.get_field(id).get_buf();

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

                    // SPH-summation gives LAB-FRAME density N (not rest-frame n)
                    Tscal N_lab = density[i];
                    N_lab       = sycl::max(N_lab, Tscal(1e-30));

                    if (has_uint && uint_ptr != nullptr) {
                        Tscal u = uint_ptr[i];
                        u       = sycl::max(u, Tscal(1e-30));

                        // Compute Lorentz factor from velocity
                        const Tvec v   = vxyz[i];
                        const Tscal v2 = sycl::dot(v, v) / c2;
                        const Tscal gam
                            = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2, Tscal{1e-10}));

                        // Rest-frame density: n = N / γ
                        const Tscal n = N_lab / gam;

                        // Relativistic EOS: P = (γ_eos - 1) * n * ε
                        // Uses REST-FRAME density, not lab-frame!
                        Tscal P = (gamma - Tscal{1}) * n * u;

                        // Specific enthalpy: H = 1 + ε/c² + P/(n·c²)
                        const Tscal H = Tscal{1} + u / c2 + P / (n * c2);

                        // Relativistic sound speed: cs² = (γ_eos - 1)(H - 1) / H
                        const Tscal cs2 = (gamma - Tscal{1}) * (H - Tscal{1}) / H;
                        Tscal cs        = sycl::sqrt(sycl::fmax(cs2, Tscal{0})) * c_speed;

                        // Clamp to reasonable values
                        P  = sycl::clamp(P, Tscal(1e-30), Tscal(1e30));
                        cs = sycl::clamp(cs, Tscal(1e-10), c_speed);

                        pressure[i]   = P;
                        soundspeed[i] = cs;
                    } else {
                        // Isothermal case (fallback)
                        Tscal cs = c_speed * Tscal{0.1}; // Default sound speed
                        Tscal P  = cs * cs * N_lab;

                        pressure[i]   = P;
                        soundspeed[i] = cs;
                    }
                });
            });

            // Complete all buffer event states
            buf_density.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            if (has_uint) {
                mpdat.get_field_buf_ref<Tscal>(iuint_interf).complete_event_state(e);
            }
            pressure_buf.complete_event_state(e);
            soundspeed_buf.complete_event_state(e);
        });
    }

    // Explicit template instantiations
    using namespace shammath;

    template class SREOSComputation<sycl::vec<double, 3>, M4>;
    template class SREOSComputation<sycl::vec<double, 3>, M6>;
    template class SREOSComputation<sycl::vec<double, 3>, M8>;
    template class SREOSComputation<sycl::vec<double, 3>, C2>;
    template class SREOSComputation<sycl::vec<double, 3>, C4>;
    template class SREOSComputation<sycl::vec<double, 3>, C6>;
    template class SREOSComputation<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::modules

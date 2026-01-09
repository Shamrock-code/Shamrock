// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NewtonianForceKernel.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Implementation of Newtonian GSPH force kernel
 *
 * Implements force computation for Newtonian hydrodynamics following
 * Cha & Whitworth (2003) "Implementations and tests of Godunov-type
 * particle hydrodynamics"
 */

#include "shambase/exception.hpp"
#include "shambackends/math.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/FieldNames.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianForceKernel.hpp"
#include "shammodels/gsph/physics/newtonian/forces.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"  // for hllc_solver
#include "shammodels/gsph/physics/newtonian/riemann/HLL.hpp"
#include "shammodels/gsph/physics/newtonian/riemann/Iterative.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::gsph::physics::newtonian {

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianForceKernel<Tvec, SPHKernel>::compute_iterative(
        const riemann::IterativeConfig &cfg) {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        using namespace shammodels::gsph;
        using Kernel = SPHKernel<Tscal>;

        PatchDataLayerLayout &pdl = scheduler_.pdl();

        const u32 ixyz   = pdl.get_field_idx<Tvec>(fields::XYZ);
        const u32 ivxyz  = pdl.get_field_idx<Tvec>(fields::VXYZ);
        const u32 iaxyz  = pdl.get_field_idx<Tvec>(fields::AXYZ);
        const u32 ihpart = pdl.get_field_idx<Tscal>(fields::HPART);

        const bool has_uint = config_.has_field_uint();
        const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>(fields::DUINT) : 0;

        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage_.ghost_layout.get());
        u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>(fields::HPART);
        u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>(fields::VXYZ);
        u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>(fields::OMEGA);
        u32 idensity_interf = ghost_layout.get_field_idx<Tscal>(computed_fields::DENSITY);

        auto &merged_xyzh                                 = storage_.merged_xyzh.get();
        shambase::DistributedData<PatchDataLayer> &mpdats = storage_.merged_patchdata_ghost.get();

        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage_.pressure);
        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage_.soundspeed);

        scheduler_.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

            sham::DeviceBuffer<Tvec> &buf_xyz
                = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
            sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
            sham::DeviceBuffer<Tscal> &buf_pressure
                = pressure_field.get_field(cur_p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = soundspeed_field.get_field(cur_p.id_patch).get_buf();

            tree::ObjectCache &pcache
                = shambase::get_check_ref(storage_.neigh_cache).get_cache(cur_p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            sham::DeviceBuffer<Tscal> &buf_density
                = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

            auto xyz          = buf_xyz.get_read_access(depends_list);
            auto axyz         = buf_axyz.get_write_access(depends_list);
            auto vxyz         = buf_vxyz.get_read_access(depends_list);
            auto hpart        = buf_hpart.get_read_access(depends_list);
            auto omega_acc    = buf_omega.get_read_access(depends_list);
            auto density_acc  = buf_density.get_read_access(depends_list);
            auto pressure_acc = buf_pressure.get_read_access(depends_list);
            auto cs_acc       = buf_cs.get_read_access(depends_list);
            auto ploop_ptrs   = pcache.get_read_access(depends_list);

            sham::DeviceBuffer<Tscal> *buf_duint_ptr = nullptr;
            Tscal *duint_acc                         = nullptr;
            if (has_uint) {
                buf_duint_ptr = &pdat.get_field_buf_ref<Tscal>(iduint);
                duint_acc     = buf_duint_ptr->get_write_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass  = config_.gpart_mass;
                const Tscal gamma  = config_.get_eos_gamma();
                const Tscal tol    = cfg.tol;
                const u32 max_iter = cfg.max_iter;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parallel_for(
                    cgh, pdat.get_obj_cnt(), "GSPH derivs iterative", [=](u64 gid) {
                        u32 id_a = (u32) gid;

                        Tvec sum_axyz  = {0, 0, 0};
                        Tscal sum_du_a = 0;

                        const Tscal h_a     = hpart[id_a];
                        const Tvec xyz_a    = xyz[id_a];
                        const Tvec vxyz_a   = vxyz[id_a];
                        const Tscal omega_a = omega_acc[id_a];

                        const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));
                        const Tscal P_a   = sycl::max(pressure_acc[id_a], Tscal(1e-30));

                        particle_looper.for_each_object(id_a, [&](u32 id_b) {
                            if (id_a == id_b)
                                return;

                            const Tvec dr    = xyz_a - xyz[id_b];
                            const Tscal rab2 = sycl::dot(dr, dr);
                            const Tscal h_b  = hpart[id_b];

                            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                                return;
                            }

                            const Tscal rab     = sycl::sqrt(rab2);
                            const Tvec vxyz_b   = vxyz[id_b];
                            const Tscal omega_b = omega_acc[id_b];

                            const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));
                            const Tscal P_b   = sycl::max(pressure_acc[id_b], Tscal(1e-30));

                            const Tscal rab_inv  = sham::inv_sat_positive(rab);
                            const Tvec r_ab_unit = dr * rab_inv;

                            const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                            const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                            auto riemann_result
                                = ::shammodels::gsph::physics::newtonian::riemann::solve<Tscal>(
                                    u_b_proj,
                                    rho_b,
                                    P_b,
                                    u_a_proj,
                                    rho_a,
                                    P_a,
                                    gamma,
                                    tol,
                                    max_iter);
                            const Tscal p_star = riemann_result.p_star;
                            const Tscal v_star = riemann_result.v_star;

                            const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                            const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                            ::shammodels::gsph::physics::newtonian::
                                add_gsph_force_contribution<Tvec, Tscal>(
                                    pmass,
                                    p_star,
                                    v_star,
                                    rho_a,
                                    rho_b,
                                    omega_a,
                                    omega_b,
                                    Fab_a,
                                    Fab_b,
                                    r_ab_unit,
                                    vxyz_a,
                                    sum_axyz,
                                    sum_du_a);
                        });

                        axyz[id_a] = sum_axyz;
                        if (duint_acc != nullptr) {
                            duint_acc[id_a] = sum_du_a;
                        }
                    });
            });

            buf_xyz.complete_event_state(e);
            buf_axyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_density.complete_event_state(e);
            buf_pressure.complete_event_state(e);
            buf_cs.complete_event_state(e);

            if (has_uint && buf_duint_ptr) {
                buf_duint_ptr->complete_event_state(e);
            }

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianForceKernel<Tvec, SPHKernel>::compute_hll(const riemann::HLLConfig &cfg) {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        using namespace shammodels::gsph;
        using Kernel = SPHKernel<Tscal>;

        PatchDataLayerLayout &pdl = scheduler_.pdl();

        const u32 iaxyz     = pdl.get_field_idx<Tvec>(fields::AXYZ);
        const bool has_uint = config_.has_field_uint();
        const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>(fields::DUINT) : 0;

        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage_.ghost_layout.get());
        u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>(fields::HPART);
        u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>(fields::VXYZ);
        u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>(fields::OMEGA);
        u32 idensity_interf = ghost_layout.get_field_idx<Tscal>(computed_fields::DENSITY);

        auto &merged_xyzh                                 = storage_.merged_xyzh.get();
        shambase::DistributedData<PatchDataLayer> &mpdats = storage_.merged_patchdata_ghost.get();

        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage_.pressure);
        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage_.soundspeed);

        scheduler_.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

            sham::DeviceBuffer<Tvec> &buf_xyz
                = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
            sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
            sham::DeviceBuffer<Tscal> &buf_pressure
                = pressure_field.get_field(cur_p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = soundspeed_field.get_field(cur_p.id_patch).get_buf();

            tree::ObjectCache &pcache
                = shambase::get_check_ref(storage_.neigh_cache).get_cache(cur_p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            sham::DeviceBuffer<Tscal> &buf_density
                = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

            auto xyz          = buf_xyz.get_read_access(depends_list);
            auto axyz         = buf_axyz.get_write_access(depends_list);
            auto vxyz         = buf_vxyz.get_read_access(depends_list);
            auto hpart        = buf_hpart.get_read_access(depends_list);
            auto omega_acc    = buf_omega.get_read_access(depends_list);
            auto density_acc  = buf_density.get_read_access(depends_list);
            auto pressure_acc = buf_pressure.get_read_access(depends_list);
            auto cs_acc       = buf_cs.get_read_access(depends_list);
            auto ploop_ptrs   = pcache.get_read_access(depends_list);

            sham::DeviceBuffer<Tscal> *buf_duint_ptr = nullptr;
            Tscal *duint_acc                         = nullptr;
            if (has_uint) {
                buf_duint_ptr = &pdat.get_field_buf_ref<Tscal>(iduint);
                duint_acc     = buf_duint_ptr->get_write_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = config_.gpart_mass;
                const Tscal gamma = config_.get_eos_gamma();

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parallel_for(cgh, pdat.get_obj_cnt(), "GSPH derivs HLL", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;

                    const Tscal h_a     = hpart[id_a];
                    const Tvec xyz_a    = xyz[id_a];
                    const Tvec vxyz_a   = vxyz[id_a];
                    const Tscal omega_a = omega_acc[id_a];

                    const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));
                    const Tscal P_a   = sycl::max(pressure_acc[id_a], Tscal(1e-30));

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        if (id_a == id_b)
                            return;

                        const Tvec dr    = xyz_a - xyz[id_b];
                        const Tscal rab2 = sycl::dot(dr, dr);
                        const Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        const Tscal rab     = sycl::sqrt(rab2);
                        const Tvec vxyz_b   = vxyz[id_b];
                        const Tscal omega_b = omega_acc[id_b];

                        const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));
                        const Tscal P_b   = sycl::max(pressure_acc[id_b], Tscal(1e-30));

                        const Tscal rab_inv  = sham::inv_sat_positive(rab);
                        const Tvec r_ab_unit = dr * rab_inv;

                        const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                        const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                        auto riemann_result
                            = ::shammodels::gsph::physics::newtonian::riemann::solve_hll<Tscal>(
                                u_b_proj, rho_b, P_b, u_a_proj, rho_a, P_a, gamma);
                        const Tscal p_star = riemann_result.p_star;
                        const Tscal v_star = riemann_result.v_star;

                        const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                        const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                        ::shammodels::gsph::physics::newtonian::
                            add_gsph_force_contribution<Tvec, Tscal>(
                                pmass,
                                p_star,
                                v_star,
                                rho_a,
                                rho_b,
                                omega_a,
                                omega_b,
                                Fab_a,
                                Fab_b,
                                r_ab_unit,
                                vxyz_a,
                                sum_axyz,
                                sum_du_a);
                    });

                    axyz[id_a] = sum_axyz;
                    if (duint_acc != nullptr) {
                        duint_acc[id_a] = sum_du_a;
                    }
                });
            });

            buf_xyz.complete_event_state(e);
            buf_axyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_density.complete_event_state(e);
            buf_pressure.complete_event_state(e);
            buf_cs.complete_event_state(e);

            if (has_uint && buf_duint_ptr) {
                buf_duint_ptr->complete_event_state(e);
            }

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianForceKernel<Tvec, SPHKernel>::compute_hllc(const riemann::HLLCConfig &cfg) {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        using namespace shammodels::gsph;
        using Kernel = SPHKernel<Tscal>;

        PatchDataLayerLayout &pdl = scheduler_.pdl();

        const u32 iaxyz     = pdl.get_field_idx<Tvec>(fields::AXYZ);
        const bool has_uint = config_.has_field_uint();
        const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>(fields::DUINT) : 0;

        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage_.ghost_layout.get());
        u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>(fields::HPART);
        u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>(fields::VXYZ);
        u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>(fields::OMEGA);
        u32 idensity_interf = ghost_layout.get_field_idx<Tscal>(computed_fields::DENSITY);

        auto &merged_xyzh                                 = storage_.merged_xyzh.get();
        shambase::DistributedData<PatchDataLayer> &mpdats = storage_.merged_patchdata_ghost.get();

        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage_.pressure);
        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage_.soundspeed);

        scheduler_.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

            sham::DeviceBuffer<Tvec> &buf_xyz
                = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
            sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
            sham::DeviceBuffer<Tscal> &buf_pressure
                = pressure_field.get_field(cur_p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &buf_cs
                = soundspeed_field.get_field(cur_p.id_patch).get_buf();

            tree::ObjectCache &pcache
                = shambase::get_check_ref(storage_.neigh_cache).get_cache(cur_p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            sham::DeviceBuffer<Tscal> &buf_density
                = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

            auto xyz          = buf_xyz.get_read_access(depends_list);
            auto axyz         = buf_axyz.get_write_access(depends_list);
            auto vxyz         = buf_vxyz.get_read_access(depends_list);
            auto hpart        = buf_hpart.get_read_access(depends_list);
            auto omega_acc    = buf_omega.get_read_access(depends_list);
            auto density_acc  = buf_density.get_read_access(depends_list);
            auto pressure_acc = buf_pressure.get_read_access(depends_list);
            auto cs_acc       = buf_cs.get_read_access(depends_list);
            auto ploop_ptrs   = pcache.get_read_access(depends_list);

            sham::DeviceBuffer<Tscal> *buf_duint_ptr = nullptr;
            Tscal *duint_acc                         = nullptr;
            if (has_uint) {
                buf_duint_ptr = &pdat.get_field_buf_ref<Tscal>(iduint);
                duint_acc     = buf_duint_ptr->get_write_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = config_.gpart_mass;
                const Tscal gamma = config_.get_eos_gamma();

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parallel_for(
                    cgh, pdat.get_obj_cnt(), "GSPH derivs HLLC", [=](u64 gid) {
                        u32 id_a = (u32) gid;

                        Tvec sum_axyz  = {0, 0, 0};
                        Tscal sum_du_a = 0;

                        const Tscal h_a     = hpart[id_a];
                        const Tvec xyz_a    = xyz[id_a];
                        const Tvec vxyz_a   = vxyz[id_a];
                        const Tscal omega_a = omega_acc[id_a];

                        const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));
                        const Tscal P_a   = sycl::max(pressure_acc[id_a], Tscal(1e-30));

                        particle_looper.for_each_object(id_a, [&](u32 id_b) {
                            if (id_a == id_b)
                                return;

                            const Tvec dr    = xyz_a - xyz[id_b];
                            const Tscal rab2 = sycl::dot(dr, dr);
                            const Tscal h_b  = hpart[id_b];

                            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                                return;
                            }

                            const Tscal rab     = sycl::sqrt(rab2);
                            const Tvec vxyz_b   = vxyz[id_b];
                            const Tscal omega_b = omega_acc[id_b];

                            const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));
                            const Tscal P_b   = sycl::max(pressure_acc[id_b], Tscal(1e-30));

                            const Tscal rab_inv  = sham::inv_sat_positive(rab);
                            const Tvec r_ab_unit = dr * rab_inv;

                            const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                            const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                            // Use HLLC approximate Riemann solver
                            auto riemann_result = ::shammodels::gsph::riemann::hllc_solver<Tscal>(
                                u_b_proj,
                                rho_b,
                                P_b,
                                u_a_proj,
                                rho_a,
                                P_a,
                                gamma);
                            const Tscal p_star = riemann_result.p_star;
                            const Tscal v_star = riemann_result.v_star;

                            const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                            const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                            ::shammodels::gsph::physics::newtonian::
                                add_gsph_force_contribution<Tvec, Tscal>(
                                    pmass,
                                    p_star,
                                    v_star,
                                    rho_a,
                                    rho_b,
                                    omega_a,
                                    omega_b,
                                    Fab_a,
                                    Fab_b,
                                    r_ab_unit,
                                    vxyz_a,
                                    sum_axyz,
                                    sum_du_a);
                        });

                        axyz[id_a] = sum_axyz;
                        if (duint_acc != nullptr) {
                            duint_acc[id_a] = sum_du_a;
                        }
                    });
            });

            buf_xyz.complete_event_state(e);
            buf_axyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_density.complete_event_state(e);
            buf_pressure.complete_event_state(e);
            buf_cs.complete_event_state(e);

            if (has_uint && buf_duint_ptr) {
                buf_duint_ptr->complete_event_state(e);
            }

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianForceKernel<Tvec, SPHKernel>::compute_roe(const riemann::RoeConfig &cfg) {
        shambase::throw_unimplemented("Roe Riemann solver not yet implemented");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Explicit template instantiations
    // ════════════════════════════════════════════════════════════════════════════

    using namespace shammath;

    template class NewtonianForceKernel<sycl::vec<double, 3>, M4>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, M6>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, M8>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, C2>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, C4>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, C6>;
    template class NewtonianForceKernel<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::newtonian

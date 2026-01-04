// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SRPhysics.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Implementation of SR-GSPH physics module
 */

#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shammodels/gsph/modules/SRPhysics.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/gsph/math/sr/primitive_recovery.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void SRPhysics<Tvec, SPHKernel>::init_fields() {
        StackEntry stack_loc{};

        using namespace shamrock::solvergraph;
        using namespace shamrock::patch;

        std::shared_ptr<Indexes<u32>> sizes = std::make_shared<Indexes<u32>>("sr_sizes", "N");
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        if (!storage.S_momentum) {
            storage.S_momentum = std::make_shared<Field<Tvec>>(1, "S_momentum", "S");
        }
        storage.S_momentum->ensure_sizes(sizes->indexes);

        if (!storage.e_energy) {
            storage.e_energy = std::make_shared<Field<Tscal>>(1, "e_energy", "e");
        }
        storage.e_energy->ensure_sizes(sizes->indexes);

        if (!storage.dS_momentum) {
            storage.dS_momentum = std::make_shared<Field<Tvec>>(1, "dS_momentum", "\\dot{S}");
        }
        storage.dS_momentum->ensure_sizes(sizes->indexes);

        if (!storage.de_energy) {
            storage.de_energy = std::make_shared<Field<Tscal>>(1, "de_energy", "\\dot{e}");
        }
        storage.de_energy->ensure_sizes(sizes->indexes);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPhysics<Tvec, SPHKernel>::init_conserved() {
        StackEntry stack_loc{};

        using namespace shamrock::patch;

        if (storage.sr_initialized) {
            return;
        }

        init_fields();

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        const Tscal c_speed   = solver_config.sr_config.get_c_speed();
        const Tscal gamma_eos = solver_config.get_eos_gamma();
        const Tscal pmass     = solver_config.gpart_mass;

        shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
        shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);
        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tvec> &buf_S    = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e   = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_rho = density_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_P   = pressure_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_vxyz, buf_rho, buf_P},
                sham::MultiRef{buf_S, buf_e},
                cnt,
                [c_speed, gamma_eos, pmass](
                    u32 i,
                    const Tvec *vxyz,
                    const Tscal *rho_sph,
                    const Tscal *P,
                    Tvec *S,
                    Tscal *e) {
                    const Tscal c2 = c_speed * c_speed;

                    const Tscal v2 = sycl::dot(vxyz[i], vxyz[i]) / c2;
                    const Tscal gamma_lor
                        = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2, Tscal{1e-10}));

                    const Tscal N_lab = sycl::fmax(rho_sph[i], Tscal{1e-30});
                    const Tscal n     = N_lab / gamma_lor;

                    const Tscal u_int = P[i] / ((gamma_eos - Tscal{1}) * n);
                    const Tscal H     = Tscal{1} + u_int / c2 + P[i] / (n * c2);

                    S[i] = vxyz[i] * (gamma_lor * H);
                    e[i] = gamma_lor * H - P[i] / (N_lab * c2);
                });
        });

        storage.sr_initialized = true;

        if (shamcomm::world_rank() == 0) {
            shamcomm::logs::raw_ln(
                "SR-GSPH: Initialized conserved variables (S, e) from primitives");
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPhysics<Tvec, SPHKernel>::do_predictor(Tscal dt) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        const Tscal half_dt   = dt / Tscal{2};
        const Tscal c_speed   = solver_config.sr_config.get_c_speed();
        const Tscal gamma_eos = solver_config.get_eos_gamma();

        shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
        shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
        shamrock::solvergraph::Field<Tvec> &dS_field = shambase::get_check_ref(storage.dS_momentum);
        shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage.de_energy);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_dS, buf_de},
                sham::MultiRef{buf_S, buf_e},
                cnt,
                [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                    S[i] += dS[i] * half_dt;
                    e[i] += de[i] * half_dt;
                });
        });

        cons2prim();

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &xyz_field  = pdat.get_field<Tvec>(ixyz);
            auto &vxyz_field = pdat.get_field<Tvec>(ivxyz);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{vxyz_field.get_buf()},
                sham::MultiRef{xyz_field.get_buf()},
                cnt,
                [dt](u32 i, const Tvec *vxyz, Tvec *xyz) {
                    xyz[i] += vxyz[i] * dt;
                });
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPhysics<Tvec, SPHKernel>::apply_corrector(Tscal dt) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;

        const Tscal half_dt = dt / Tscal{2};

        shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
        shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
        shamrock::solvergraph::Field<Tvec> &dS_field = shambase::get_check_ref(storage.dS_momentum);
        shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage.de_energy);

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_dS, buf_de},
                sham::MultiRef{buf_S, buf_e},
                cnt,
                [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                    S[i] += dS[i] * half_dt;
                    e[i] += de[i] * half_dt;
                });
        });

        cons2prim();
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRPhysics<Tvec, SPHKernel>::cons2prim() {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        namespace sr = shammodels::gsph::sr;

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const bool has_uint       = solver_config.has_field_uint();
        const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

        const Tscal c_speed   = solver_config.sr_config.get_c_speed();
        const Tscal gamma_eos = solver_config.get_eos_gamma();
        const Tscal pmass     = solver_config.gpart_mass;

        shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
        shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);
        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tvec> &buf_S    = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e   = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_rho = density_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_P   = pressure_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            if (has_uint) {
                sham::DeviceBuffer<Tscal> &buf_uint = pdat.get_field_buf_ref<Tscal>(iuint);

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{buf_S, buf_e, buf_rho},
                    sham::MultiRef{buf_vxyz, buf_P, buf_uint},
                    cnt,
                    [c_speed, gamma_eos](
                        u32 i,
                        const Tvec *S,
                        const Tscal *e,
                        const Tscal *rho_sph,
                        Tvec *vxyz,
                        Tscal *P,
                        Tscal *uint_out) {
                        const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                        const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                        const Tscal N = sycl::fmax(rho_sph[i], Tscal{1e-30});

                        sr::SRPrimitiveVars<Tscal> prim = sr::conserved_to_primitive<Tscal>(
                            S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                        vxyz[i] = S_dir * prim.vel_normal * c_speed;
                        P[i]    = sycl::fmax(prim.pressure, Tscal{1e-10});

                        uint_out[i] = P[i] / ((gamma_eos - Tscal{1}) * N);
                    });
            } else {
                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{buf_S, buf_e, buf_rho},
                    sham::MultiRef{buf_vxyz, buf_P},
                    cnt,
                    [c_speed, gamma_eos](
                        u32 i,
                        const Tvec *S,
                        const Tscal *e,
                        const Tscal *rho_sph,
                        Tvec *vxyz,
                        Tscal *P) {
                        const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                        const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                        const Tscal N = sycl::fmax(rho_sph[i], Tscal{1e-30});

                        sr::SRPrimitiveVars<Tscal> prim = sr::conserved_to_primitive<Tscal>(
                            S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                        vxyz[i] = S_dir * prim.vel_normal * c_speed;
                        P[i]    = sycl::fmax(prim.pressure, Tscal{1e-10});
                    });
            }
        });
    }

    // Explicit template instantiations
    using namespace shammath;

    template class SRPhysics<sycl::vec<double, 3>, M4>;
    template class SRPhysics<sycl::vec<double, 3>, M6>;
    template class SRPhysics<sycl::vec<double, 3>, M8>;
    template class SRPhysics<sycl::vec<double, 3>, C2>;
    template class SRPhysics<sycl::vec<double, 3>, C4>;
    template class SRPhysics<sycl::vec<double, 3>, C6>;
    template class SRPhysics<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::modules

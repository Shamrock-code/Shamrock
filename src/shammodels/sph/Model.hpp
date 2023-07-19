// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/exchanges.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/generic/setup/generators.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::sph {

    /**
     * @brief The shamrock SPH model
     *
     * @tparam Tvec
     * @tparam SPHKernel
     */
    template<class Tvec, template<class> class SPHKernel>
    class Model {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Solver = Solver<Tvec, SPHKernel>;
        // using SolverConfig = typename Solver::Config;

        ShamrockCtx &ctx;

        Solver solver;

        // SolverConfig sconfig;

        Model(ShamrockCtx &ctx) : ctx(ctx), solver(ctx){};

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// setup function
        ////////////////////////////////////////////////////////////////////////////////////////////

        void init_scheduler(u32 crit_split, u32 crit_merge);

        template<std::enable_if_t<dim == 3, int> = 0>
        inline Tvec get_box_dim_fcc_3d(Tscal dr, u32 xcnt, u32 ycnt, u32 zcnt) {
            return generic::setup::generators::get_box_dim(dr, xcnt, ycnt, zcnt);
        }

        inline void set_cfl_cour(Tscal cfl_cour) { solver.cfl_cour = cfl_cour; }
        inline void set_cfl_force(Tscal cfl_force) { solver.cfl_force = cfl_force; }
        inline void set_particle_mass(Tscal gpart_mass) { solver.gpart_mass = gpart_mass; }
        inline void set_eos_gamma(Tscal eos_gamma) { solver.eos_gamma = eos_gamma; }

        inline void resize_simulation_box(std::pair<Tvec, Tvec> box) {
            ctx.set_coord_domain_bound({box.first, box.second});
        }

        u64 get_total_part_count();

        f64 total_mass_to_part_mass(f64 totmass);


        std::pair<Tvec, Tvec> get_ideal_fcc_box(Tscal dr, std::pair<Tvec, Tvec> box);


        void add_cube_fcc_3d(Tscal dr, std::pair<Tvec, Tvec> _box);

        inline void add_sink(Tscal mass, Tvec pos, Tvec velocity){
            if(solver.storage.sinks.is_empty()){
                solver.storage.sinks.set({});
            }

            solver.storage.sinks.get().push_back({
                pos,velocity,{},{},mass,{}
            });
        }

        inline void set_units(shamrock::UnitSystem<Tscal> unit_sys){
            solver.solver_config.unit_sys = unit_sys;
        }

        template<std::enable_if_t<dim == 3, int> = 0>
        inline void add_cube_disc_3d(Tvec center,
                                     u32 Npart,
                                     Tscal p,
                                     Tscal rho_0,
                                     Tscal m,
                                     Tscal r_in,
                                     Tscal r_out,
                                     Tscal q) {

            using namespace shamrock::patch;

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

            std::string log = "";  

            sched.for_each_local_patchdata([&](const Patch ptch, PatchData &pdat) {
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(ptch);

                std::vector<Tvec> vec_acc;
                generic::setup::generators::add_disc(
                    Npart, p, rho_0, m, r_in, r_out, q, [&](Tvec r, Tscal h) {
                        vec_acc.push_back(r + center);
                    });

                log += shambase::format("\n    patch id={}, add N={} particles", ptch.id_patch, vec_acc.size());

                PatchData tmp(sched.pdl);
                tmp.resize(vec_acc.size());
                tmp.fields_raz();

                {
                    u32 len = vec_acc.size();
                    PatchDataField<Tvec> &f =
                        tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                    sycl::buffer<Tvec> buf(vec_acc.data(), len);
                    f.override(buf, len);
                }

                {
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                    f.override(0.01);
                }

                pdat.insert_elements(tmp);
            });

            std::string log_gathered = "";
            shamalgs::collective::gather_str(log, log_gathered);

            if(shamsys::instance::world_rank == 0) {
                logger::info_ln("Model", "Push particles : ", log_gathered);
            }

            sched.scheduler_step(false, false);


            {
                auto [m, M] = sched.get_box_tranform<Tvec>();


                SerialPatchTree<Tvec> sptree(sched.patch_tree,
                                             sched.get_sim_box().get_patch_transform<Tvec>());

                // sptree.print_status();

                shamrock::ReattributeDataUtility reatrib(sched);

                sptree.attach_buf();
                // reatribute_particles(sched, sptree, periodic_mode);

                reatrib.reatribute_patch_objects(sptree, "xyz");
            }

            sched.check_patchdata_locality_corectness();

            sched.scheduler_step(true, true);

            log = "";            
            sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
                log += shambase::format("\n    patch id={}, N={} particles", p.id_patch, pdat.get_obj_cnt());
            });

            log_gathered = "";
            shamalgs::collective::gather_str(log, log_gathered);

            if(shamsys::instance::world_rank == 0) logger::info_ln("Model", "current particle counts : ", log_gathered);
        }

        template<class T>
        inline void set_value_in_a_box(std::string field_name, T val, std::pair<Tvec, Tvec> box) {
            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchData &pdat) {

                    PatchDataField<Tvec> &xyz =
                        pdat.template get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f =
                        pdat.template get_field<T>(sched.pdl.get_field_idx<T>(field_name));

                    {
                        auto &buf = shambase::get_check_ref(f.get_buf());
                        sycl::host_accessor acc{buf};

                        auto &buf_xyz = shambase::get_check_ref(xyz.get_buf());
                        sycl::host_accessor acc_xyz{buf_xyz};

                        for (u32 i = 0; i < f.size(); i++) {
                            Tvec r = acc_xyz[i];

                            if (BBAA::is_coord_in_range(r, std::get<0>(box), std::get<1>(box))) {
                                acc[i] = val;
                            }
                        }
                    }
                });
        }

        template<class T>
        inline void set_value_in_sphere(std::string field_name, T val, Tvec center, Tscal radius) {
            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchData &pdat) {
                    PatchDataField<Tvec> &xyz =
                        pdat.template get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f =
                        pdat.template get_field<T>(sched.pdl.get_field_idx<T>(field_name));

                    Tscal r2 = radius * radius;
                    {
                        auto &buf = shambase::get_check_ref(f.get_buf());
                        sycl::host_accessor acc{buf};

                        auto &buf_xyz = shambase::get_check_ref(xyz.get_buf());
                        sycl::host_accessor acc_xyz{buf_xyz};

                        for (u32 i = 0; i < f.size(); i++) {
                            Tvec dr = acc_xyz[i] - center;

                            if (sycl::dot(dr, dr) < r2) {
                                acc[i] = val;
                            }
                        }
                    }
                });
        }

        template<class T>
        inline void add_kernel_value(std::string field_name, T val, Tvec center, Tscal h_ker) {
            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchData &pdat) {
                    PatchDataField<Tvec> &xyz =
                        pdat.template get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f =
                        pdat.template get_field<T>(sched.pdl.get_field_idx<T>(field_name));

                    {
                        auto &buf = shambase::get_check_ref(f.get_buf());
                        sycl::host_accessor acc{buf};

                        auto &buf_xyz = shambase::get_check_ref(xyz.get_buf());
                        sycl::host_accessor acc_xyz{buf_xyz};

                        for (u32 i = 0; i < f.size(); i++) {
                            Tvec dr = acc_xyz[i] - center;

                            Tscal r = sycl::length(dr);

                            acc[i] += val*Kernel::W(r,h_ker);
                        }
                    }
                });
        }

        template<class T>
        inline T get_sum(std::string name) {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            T sum                 = shambase::VectorProperties<T>::get_zero();

            StackEntry stack_loc{};
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchData &pdat) {
                    PatchDataField<T> &xyz =
                        pdat.template get_field<T>(sched.pdl.get_field_idx<T>(name));

                    sum += xyz.compute_sum();
                });

            return shamalgs::collective::allreduce_sum(sum);
        }

        Tvec get_closest_part_to(Tvec pos);

        // inline void enable_barotropic_mode(){
        //     sconfig.enable_barotropic();
        // }
        //
        // inline void switch_internal_energy_mode(std::string name){
        //     sconfig.switch_internal_energy_mode(name);
        // }

        inline void set_solver_config(typename Solver::Config cfg) { solver.solver_config = cfg; }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// analysis utilities
        ////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// I/O
        ////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// Simulation control
        ////////////////////////////////////////////////////////////////////////////////////////////

        f64
        evolve_once(f64 t_curr, f64 dt_input, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id);
    };

} // namespace shammodels
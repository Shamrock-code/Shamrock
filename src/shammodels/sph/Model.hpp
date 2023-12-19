// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Model.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shamalgs/collective/exchanges.hpp"
#include "shambase/string.hpp"
#include "shambase/Constants.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamcomm/collectives.hpp"
#include "shammodels/generic/setup/generators.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <hipSYCL/sycl/libkernel/builtins.hpp>
#include <vector>

#include <pybind11/functional.h>

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
        using SolverConfig = typename Solver::Config;
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

        inline void set_cfl_cour(Tscal cfl_cour) { solver.solver_config.cfl_cour = cfl_cour; }
        inline void set_cfl_force(Tscal cfl_force) { solver.solver_config.cfl_force = cfl_force; }
        inline void set_particle_mass(Tscal gpart_mass) { solver.solver_config.gpart_mass = gpart_mass; }

        inline void resize_simulation_box(std::pair<Tvec, Tvec> box) {
            ctx.set_coord_domain_bound({box.first, box.second});
        }


        SolverConfig gen_config_from_phantom_dump(PhantomDump & phdump, bool bypass_error);
        void init_from_phantom_dump(PhantomDump & phdump);
        PhantomDump make_phantom_dump();

        u64 get_total_part_count();

        f64 total_mass_to_part_mass(f64 totmass);


        std::pair<Tvec, Tvec> get_ideal_fcc_box(Tscal dr, std::pair<Tvec, Tvec> box);

        Tscal get_hfact(){
            return Kernel::hfactd;
        }
        
        Tscal rho_h(Tscal h){
            return shamrock::sph::rho_h(solver.solver_config.gpart_mass, h, Kernel::hfactd);
        }

        void add_cube_fcc_3d(Tscal dr, std::pair<Tvec, Tvec> _box);

        inline void add_sink(Tscal mass, Tvec pos, Tvec velocity, Tscal accretion_radius){
            if(solver.storage.sinks.is_empty()){
                solver.storage.sinks.set({});
            }

            logger::debug_ln("SPH", "add sink :",mass,pos,velocity,accretion_radius);

            solver.storage.sinks.get().push_back({
                pos,velocity,{},{},mass,{},accretion_radius
            });
        }

        template<class T>
        inline void set_field_value_lambda(std::string field_name, const std::function<T(Tvec)> pos_to_val){

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

                            acc[i] = pos_to_val(r);
                            
                        }
                    }
                });

        }

        /**
         * @brief Add a disc distribution
         * 
         * @param center position of the center of the disc
         * @param central_mass star mass
         * @param Npart number of particles
         * @param r_in inner radius
         * @param r_out outer radius
         * @param disc_mass disc mass
         * @param p density power profile
         * @param H_r_in inner radisu H/r
         * @param q soundspeed power profile
         * @return Tscal 
         */
        template<std::enable_if_t<dim == 3, int> = 0>
        inline Tscal add_disc_3d(
                Tvec center, 
                Tscal central_mass,
                u32 Npart,
                Tscal r_in,
                Tscal r_out,
                Tscal disc_mass,
                Tscal p,
                Tscal H_r_in,
                Tscal q,
                bool do_warp=false,
                Tscal posangle=0 ,
                Tscal incl=30.,
                Tscal Rwarp=5.,
                Tscal Hwarp=1.){

            Tscal G = solver.solver_config.get_constant_G();

            Tscal eos_gamma;
            using Config = SolverConfig;
            using SolverConfigEOS     = typename Config::EOSConfig;
            using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
            if (SolverEOS_Adiabatic *eos_config =
                std::get_if<SolverEOS_Adiabatic>(&solver.solver_config.eos_config.config)) {

                eos_gamma = eos_config->gamma;

            } else {
                //dirty hack for disc setup in locally isothermal
                eos_gamma = 2;
                //shambase::throw_unimplemented();
            }

            using Out = generic::setup::generators::DiscOutput<Tscal>;

            auto sigma_profile = [=](Tscal r){
                // we setup with an adimensional mass since it is monte carlo
                constexpr Tscal sigma_0 = 1;
                return sigma_0*sycl::pow(r/r_in, -p);
            };

            auto cs_law = [=](Tscal r){
                return sycl::pow(r/r_in, -q);
            };

            auto rot_profile = [=](Tscal r){
                return sycl::sqrt(G * central_mass/r);
            };

            Tscal cs_in = H_r_in*rot_profile(r_in);
            auto cs_profile = [&](Tscal r){
                return cs_law(r)*cs_in;
            };

            std::vector<Out> part_list ;

            generic::setup::generators::add_disc2<Tscal>(Npart, r_in, r_out, 
                [&](Tscal r){return sigma_profile(r);}, 
                [&](Tscal r){return cs_profile(r);}, 
                [&](Tscal r){return rot_profile(r);}, 
                [&](Out out){
                    part_list.push_back(out);
                }
            );

            Tscal part_mass = disc_mass/Npart;


            using namespace shamrock::patch;

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

            std::string log = "";  

            sched.for_each_local_patchdata([&](const Patch ptch, PatchData &pdat) {
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(ptch);

                std::vector<Tvec> vec_pos;
                std::vector<Tvec> vec_vel;
                std::vector<Tscal> vec_u;
                std::vector<Tscal> vec_h;

                std::vector<Tscal> vec_cs;

                Tscal G = solver.solver_config.get_constant_G();

                for(Out o : part_list){
                    vec_pos.push_back(o.pos + center);
                    vec_vel.push_back(o.velocity);

                    //for disc with P = \rho u (/gamma - 1)
                    //the scaleheight : H = \sqrt{u (\gamma -1)}/\Omega_K
                    //therefor the effective soundspeed is : \sqrt{(\gamma -1)u}
                    //whereas the real one is \sqrt{(\gamma -1)\gamma u}
                    vec_u.push_back(o.cs*o.cs/(/*solver.eos_gamma * */ (eos_gamma - 1)));
                    vec_h.push_back(shamrock::sph::h_rho(part_mass, o.rho, Kernel::hfactd));
                    vec_cs.push_back(o.cs);
                }

                log += shambase::format("\n    patch id={}, add N={} particles", ptch.id_patch, vec_pos.size());

                if (do_warp) {
                warp_disc<Tscal>(vec_pos, vec_vel, posangle, incl, Rwarp, Hwarp);
                }

                PatchData tmp(sched.pdl);
                tmp.resize(vec_pos.size());
                tmp.fields_raz();

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tvec> &f =
                        tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                    sycl::buffer<Tvec> buf(vec_pos.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                    sycl::buffer<Tscal> buf(vec_h.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("uint"));
                    sycl::buffer<Tscal> buf(vec_u.data(), len);
                    f.override(buf, len);
                }

                if(solver.solver_config.is_eos_locally_isothermal()){
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("soundspeed"));
                    sycl::buffer<Tscal> buf(vec_cs.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tvec> &f =
                        tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
                    sycl::buffer<Tvec> buf(vec_vel.data(), len);
                    f.override(buf, len);
                }

                pdat.insert_elements(tmp);
            });

            std::string log_gathered = "";
            shamcomm::gather_str(log, log_gathered);

            if(shamcomm::world_rank() == 0) {
                logger::info_ln("Model", "Push particles : ", log_gathered);
            }
            
            modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();


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
            shamcomm::gather_str(log, log_gathered);

            if(shamcomm::world_rank() == 0) logger::info_ln("Model", "current particle counts : ", log_gathered);
            return part_mass;
        }

        template<std::enable_if_t<dim == 3, int> = 0>
        inline void add_cube_disc_3d(Tvec center,
                                     u32 Npart,
                                     Tscal p,
                                     Tscal rho_0,
                                     Tscal m,
                                     Tscal r_in,
                                     Tscal r_out,
                                     Tscal q,
                                     Tscal cmass) {

            Tscal eos_gamma;
            using Config = SolverConfig;
            using SolverConfigEOS     = typename Config::EOSConfig;
            using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
            if (SolverEOS_Adiabatic *eos_config =
                std::get_if<SolverEOS_Adiabatic>(&solver.solver_config.eos_config.config)) {

                eos_gamma = eos_config->gamma;

            } else {
                shambase::throw_unimplemented();
            }

            auto cs = [&](Tscal u){
                return sycl::sqrt(eos_gamma * (eos_gamma - 1) * u);
            };

            auto U = [&](Tscal cs){
                return cs*cs/(eos_gamma * (eos_gamma - 1));
            };


            using namespace shamrock::patch;

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

            std::string log = "";  

            sched.for_each_local_patchdata([&](const Patch ptch, PatchData &pdat) {
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(ptch);

                std::vector<Tvec> vec_acc;
                std::vector<Tvec> vec_vel;
                std::vector<Tscal> vec_u;

                Tscal G = solver.solver_config.get_constant_G();




                generic::setup::generators::add_disc(
                    Npart, p, rho_0, m, r_in, r_out, q, [&](Tvec r, Tscal h) {
                        vec_acc.push_back(r + center);

                        Tscal R = sycl::length(r);

                        Tscal V = sycl::sqrt(G * cmass/R);

                        Tvec etheta= {-r.z(),0, r.x()};
                        etheta /= sycl::length(etheta);

                        vec_vel.push_back(V*etheta);

                        Tscal cs0 = 1;
                        Tscal cs = cs0*sycl::pow(R,-q);

                        vec_u.push_back(U(cs));
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

                {
                    u32 len = vec_acc.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("uint"));
                    sycl::buffer<Tscal> buf(vec_u.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_acc.size();
                    PatchDataField<Tvec> &f =
                        tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
                    sycl::buffer<Tvec> buf(vec_vel.data(), len);
                    f.override(buf, len);
                }

                pdat.insert_elements(tmp);
            });

            std::string log_gathered = "";
            shamcomm::gather_str(log, log_gathered);

            if(shamcomm::world_rank() == 0) {
                logger::info_ln("Model", "Push particles : ", log_gathered);
            }

            modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();

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
            shamcomm::gather_str(log, log_gathered);

            if(shamcomm::world_rank() == 0) logger::info_ln("Model", "current particle counts : ", log_gathered);
        }

        void push_particle(std::vector<Tvec> & part_pos_insert, std::vector<Tscal> & part_hpart_insert, std::vector<Tscal> &part_u_insert);

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

                            acc[i] += val*Kernel::W_3d(r,h_ker);
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
    
        private:
        void add_pdat_to_phantom_block(PhantomDumpBlock & block, shamrock::patch::PatchData & pdat);

        template<class T>
        inline void warp_disc(Tvec pos, Tvec vel, Tscal posangle, Tscal incl, Tscal Rwarp, Tscal Hwarp) {
            Tvec k = Tvec(-std::sin(posangle), 0., cos(posangle));
            Tscal inc;
            Tscal psi = 0.;
            u32 len = pos.size();

            for (i32 i=0; i < len; i++){
                Tscal R = sycl::sqrt(sycl::dot(pos(i), pos(i)));
                if (R < Rwarp - Hwarp){
                    inc = 0.;
                }
                else if (R < Rwarp + 3. * Hwarp) {
                    inc = sycl::asin(0.5 * (1. + sycl::sin(shambase::constants::pi<T> / (2. * Hwarp) * (R - Rwarp))) * sycl::sin(incl));
                    psi = shambase::constants::pi<T> * Rwarp / (4. * Hwarp) * sycl::sin(incl) / sycl::sqrt(1. - (0.5 * sycl::pow(sycl::sin(incl)), 2));
                    Tscal psimax = sycl::max(psimax, psi);
                }
                else{
                    inc = 0.;
            }

            //rotation position and velocity

            rotate_vector<T>(pos, k, inc);
            rotate_vector<T>(vel, k, inc);
            }
    
           
        }

        template<class T>
        inline void rotate_vector(Tvec u, Tvec v, Tscal theta){
            // normalize the reference direction
            Tscal vunit = v / sycl::sqrt(sycl::dot(v, v));
            Tvec w = sycl::cross(vunit, u);
            // Rodrigues' rotation formula
            u = u * sycl::cos(theta) + w * sycl::sin(theta) + vunit * sycl::dot(vunit, u) * (1. - sycl::cos(theta));
        }
    };



} // namespace shammodels
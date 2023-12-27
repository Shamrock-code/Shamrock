// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Solver.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "SolverConfig.hpp"
#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SolverLog.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shamrock/tree/TreeTraversalCache.hpp"
#include "shamsys/legacy/log.hpp"
#include <memory>
#include <stdexcept>
#include <variant>
namespace shammodels::sph {

    

    /**
     * @brief The shamrock SPH model
     *
     * @tparam Tvec
     * @tparam SPHKernel
     */
    template<class Tvec, template<class> class SPHKernel>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;
        using u_morton           = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        SolverStorage<Tvec, u_morton> storage {};

        Config solver_config;
        SolverLog solve_logs;

        static constexpr Tscal htol_up_tol  = 1.1;
        static constexpr Tscal htol_up_iter = 1.1;


        inline void init_required_fields() {
            context.pdata_layout_add_field<Tvec>("xyz", 1);
            context.pdata_layout_add_field<Tvec>("vxyz", 1);
            context.pdata_layout_add_field<Tvec>("axyz", 1);
            context.pdata_layout_add_field<Tvec>("axyz_ext", 1);
            context.pdata_layout_add_field<Tscal>("hpart", 1);

            if (solver_config.has_field_uint()) {
                context.pdata_layout_add_field<Tscal>("uint", 1);
                context.pdata_layout_add_field<Tscal>("duint", 1);
            }

            if (solver_config.has_field_alphaAV()) {
                context.pdata_layout_add_field<Tscal>("alpha_AV", 1);
            }

            if (solver_config.has_field_divv()) {
                context.pdata_layout_add_field<Tscal>("divv", 1);
            }

            if (solver_config.has_field_dtdivv()) {
                context.pdata_layout_add_field<Tscal>("dtdivv", 1);
            }

            if (solver_config.has_field_curlv()) {
                context.pdata_layout_add_field<Tvec>("curlv", 1);
            }

            if(solver_config.has_field_soundspeed()){

                // this should not be needed idealy, but we need the pressure on the ghosts and 
                // we don't want to communicate it as it can be recomputed from the other fields
                // hence we copy the soundspeed at the end of the step to a field in the patchdata
                context.pdata_layout_add_field<Tscal>("soundspeed", 1);
            }
        }

        // serial patch tree control
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { storage.serial_patch_tree.reset(); }

        // interface_control
        using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        inline void gen_ghost_handler(Tscal time_val) {

            using CfgClass = sph::BasicSPHGhostHandlerConfig<Tvec>;
            using BCConfig = typename CfgClass::Variant;

            using BCFree = typename CfgClass::Free;
            using BCPeriodic = typename CfgClass::Periodic;
            using BCShearingPeriodic = typename CfgClass::ShearingPeriodic;

            using SolverConfigBC = typename Config::BCConfig;
            using SolverBCFree = typename SolverConfigBC::Free;
            using SolverBCPeriodic = typename SolverConfigBC::Periodic;
            using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

            //boundary condition selections
            if(SolverBCFree* c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)){
                storage.ghost_handler.set(GhostHandle{scheduler(),BCFree{}});
            }else if(SolverBCPeriodic* c = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)){
                storage.ghost_handler.set(GhostHandle{scheduler(),BCPeriodic{}});
            }else if(SolverBCShearingPeriodic* c = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)){
                storage.ghost_handler.set(GhostHandle{scheduler(),BCShearingPeriodic{
                    c->shear_base, c->shear_dir, c->shear_speed*time_val, c->shear_speed
                }});
            }

            
        }
        inline void reset_ghost_handler() { storage.ghost_handler.reset(); }

        void build_ghost_cache();
        void clear_ghost_cache();


        void merge_position_ghost();

        // trees
        using RTree = RadixTree<u_morton, Tvec>;
        void build_merged_pos_trees();
        void clear_merged_pos_trees();

        void compute_presteps_rint();
        void reset_presteps_rint();

        void start_neighbors_cache();
        void reset_neighbors_cache();
        


        void sph_prestep(Tscal time_val, Tscal dt);

        void apply_position_boundary(Tscal time_val);

        void do_predictor_leapfrog(Tscal dt);

        void update_artificial_viscosity(Tscal dt);

        void init_ghost_layout();

        void communicate_merge_ghosts_fields();
        void reset_merge_ghosts_fields();

        void compute_eos_fields();
        void reset_eos_fields();

        void prepare_corrector();
        void update_derivs();
        /**
         * @brief 
         * 
         * @return true corrector is converged
         * @return false corrector is not converged
         */
        bool apply_corrector(Tscal dt, u64 Npart_all);



        Solver(ShamrockCtx &context) : context(context) {}

        void vtk_do_dump(std::string filename, bool add_patch_world_id);

        void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename){
            solver_config.set_debug_dump(_do_debug_dump, _debug_dump_filename);
        }

        void evolve_once();

        Tscal evolve_once_time_expl(Tscal t_current,Tscal dt_input){
                            solver_config.set_time(t_current);
                            solver_config.set_next_dt(dt_input);
                            evolve_once();
                            return solver_config.get_dt_sph();
                          }

        
        inline bool evolve_until(Tscal target_time,i32 niter_max){
            auto step = [&](){
                Tscal dt = solver_config.get_dt_sph();
                Tscal t = solver_config.get_time();

                if(t > target_time){
                    throw shambase::make_except_with_loc<std::invalid_argument>("the target time is higher than the current time");
                }

                if(t + dt > target_time){
                    solver_config.set_next_dt(target_time - t);
                }
                evolve_once();
            };

            i32 iter_count = 0;
            
            while(solver_config.get_time() < target_time){
                step();
                iter_count++;

                if((iter_count >= niter_max) && (niter_max != -1)){
                    logger::info_ln("SPH","stopping evolve until because of niter =",iter_count);
                    return false;
                }
            }

            return true;
            
        }

    };

} // namespace shammodels::sph
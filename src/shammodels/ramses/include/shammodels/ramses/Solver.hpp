// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Solver.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shammodels::basegodunov {
    template<class Tvec, class TgridVec>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u64;
        using Config   = SolverConfig<Tvec, TgridVec>;

        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        SolverStorage<Tvec, TgridVec, u_morton> storage{};

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
            context.pdata_layout_add_field<Tscal>("rho", AMRBlock::block_size);
            context.pdata_layout_add_field<Tvec>("rhovel", AMRBlock::block_size);
            context.pdata_layout_add_field<Tscal>("rhoetot", AMRBlock::block_size);

            if (solver_config.is_dust_on()) {
                u32 ndust = solver_config.dust_config.ndust;

                context.pdata_layout_add_field<Tscal>("rho_dust", (ndust * AMRBlock::block_size));
                context.pdata_layout_add_field<Tvec>("rhovel_dust", (ndust * AMRBlock::block_size));
            }

            if (solver_config.is_gravity_on()) {
                context.pdata_layout_add_field<Tscal>("phi", AMRBlock::block_size);
          }
            if (solver_config.is_gas_passive_scalar_on()) {
                u32 npscal_gas = solver_config.npscal_gas_config.npscal_gas;
                context.pdata_layout_add_field<Tscal>(
                    "rho_gas_pscal", (npscal_gas * AMRBlock::block_size)); 
        }

        Solver(ShamrockCtx &context) : context(context) {}

        void do_debug_vtk_dump(std::string filename);

        inline void print_timestep_logs() {
            if (shamcomm::world_rank() == 0) {
                // logger::info_ln("Godunov", "iteration since start :",
                // solve_logs.get_iteration_count());
                logger::info_ln(
                    "Godunov", "time since start :", shambase::details::get_wtime(), "(s)");
            }
        }

        void evolve_once();

        inline Tscal evolve_once_time_expl(Tscal t_current, Tscal dt_input) {
            solver_config.set_time(t_current);
            solver_config.set_next_dt(dt_input);
            evolve_once();
            return solver_config.get_dt();
        }

        inline bool evolve_until(Tscal target_time, i32 niter_max) {
            auto step = [&]() {
                Tscal dt = solver_config.get_dt();
                Tscal t  = solver_config.get_time();

                if (t > target_time) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "the target time is higher than the current time");
                }

                if (t + dt > target_time) {
                    solver_config.set_next_dt(target_time - t);
                }
                evolve_once();
            };

            i32 iter_count = 0;

            while (solver_config.get_time() < target_time) {
                step();
                iter_count++;

                if ((iter_count >= niter_max) && (niter_max != -1)) {
                    logger::info_ln("SPH", "stopping evolve until because of niter =", iter_count);
                    return false;
                }
            }

            print_timestep_logs();

            return true;
        }

        void init_solver_graph();
    };

} // namespace shammodels::basegodunov

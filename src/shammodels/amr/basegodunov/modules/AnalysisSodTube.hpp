// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AnalysisSodTube.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shamphys/SodTube.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class AnalysisSodTube {
        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config   = SolverConfig<Tvec, TgridVec>;
        using Storage  = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton = u64;
        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        shamphys::SodTube solution;
        Tvec direction;
        Tscal time_val;
        Tscal x_ref;        // shock centered on x_ref
        Tscal x_min, x_max; // check only between [x_min, x_max ]

        AnalysisSodTube(
            ShamrockCtx &context,
            Config &solver_config,
            Storage &storage,
            shamphys::SodTube &solution,
            Tvec direction,
            Tscal time_val,
            Tscal x_ref,
            Tscal x_min,
            Tscal x_max)
            : context(context), solver_config(solver_config), storage(storage), solution(solution),
              direction(direction), time_val(time_val), x_ref(x_ref), x_min(x_min), x_max(x_max) {}

        struct field_val {
            Tscal rho;
            Tvec v;
            Tscal P;
        };

        field_val compute_L2_dist();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules

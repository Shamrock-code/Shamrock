// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ExternalForces.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class ExternalForces {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        ExternalForces(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief is ran once per timestep, it computes the forces that are independant of velocity
         *
         */
        void compute_ext_forces_indep_v();

        /**
         * @brief add external forces to the particle acceleration, note that forces dependant on
         * velocity shlould be added here
         *
         */
        void add_ext_forces();

        void point_mass_accrete_particles();

        private:


        using SolverConfigExtForce = typename Config::ExtForceConfig;
        using EF_PointMass         = typename SolverConfigExtForce::PointMass;
        using EF_LenseThirring    = typename SolverConfigExtForce::LenseThirring;
        using EF_ShearingBoxForce    = typename SolverConfigExtForce::ShearingBoxForce;

        
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
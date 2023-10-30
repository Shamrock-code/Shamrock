// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SourceStep.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    template<class Tvec, class TgridVec>
    class SourceStep {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec,TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SourceStep(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief compute general forces (pressure + external and store them into `SolverStorage::forces`)
         * \f[
         *   \frac{u_i^{n+1} - u_i^{n}}{\Delta t} = \underbrace{-\frac{\partial_i p^n}{\rho^n} 
         *     + f_{\text{ext},i}}_{ f_{\rm gen} }
         * \f]
         */
        void compute_forces();

        

        /**
         * @brief apply the generalized forces
         * \f[
         *   \frac{u_i^{n+1} - u_i^{n}}{\Delta t} = \underbrace{-\frac{\partial_i p^n}{\rho^n} 
         *     + f_{\text{ext},i}}_{ f_{\rm gen} }
         * \f]
         */
        void apply_force(Tscal dt);
        
        /**
         * @brief Compute the values of the artificial viscosity terms
         * (\cite Zeus2d_main equations 33,34)
         * 
         * \f{eqnarray*}{
         *     q^x_{i,j,k} &=& C^{\rm AV} \rho_{i,j} (v^x_{i+1,j,k} - v^x_{i,j,k}) \\ 
         *     q^y_{i,j,k} &=& C^{\rm AV} \rho_{i,j} (v^y_{i,j+1,k} - v^y_{i,j,k}) \\ 
         *     q^z_{i,j,k} &=& C^{\rm AV} \rho_{i,j} (v^z_{i,j,k+1} - v^z_{i,j,k})
         * \f}
         */
        void compute_AV();

        void apply_AV(Tscal dt);

        void compute_div_v();

        void update_eint_eos(Tscal dt);
        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

    };
} // namespace shammodels::zeus::modules
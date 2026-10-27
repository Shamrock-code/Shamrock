// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GSPHSetup.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shammodels/gsph/modules/setup/IGSPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <memory>

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GSPHSetup {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        GSPHSetup(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void apply_setup(SetupNodePtr setup, std::optional<u32> insert_step = std::nullopt);

        std::shared_ptr<IGSPHSetupNode> make_generator_disc_mc(
            Tscal part_mass,
            Tscal disc_mass,
            Tscal r_in,
            Tscal r_out,
            std::function<Tscal(Tscal)> sigma_profile,
            std::function<Tscal(Tscal)> H_profile,
            std::function<Tvec(Tvec)> vel_profile,
            std::function<Tscal(Tvec)> cs_profile,
            std::mt19937_64 eng,
            Tscal init_h_factor);

        std::shared_ptr<IGSPHSetupNode> make_generator_from_context(ShamrockCtx &context_other);

        std::shared_ptr<IGSPHSetupNode> make_combiner_add(
            SetupNodePtr parent1, SetupNodePtr parent2);

        std::shared_ptr<IGSPHSetupNode> make_modifier_warp_disc(
            SetupNodePtr parent, Tscal Rwarp, Tscal Hwarp, Tscal inclination, Tscal posangle);

        std::shared_ptr<IGSPHSetupNode> make_modifier_custom_warp(
            SetupNodePtr parent,
            std::function<Tscal(Tscal)> inc_profile,
            std::function<Tscal(Tscal)> psi_profile,
            std::function<Tvec(Tscal)> k_profile);

        std::shared_ptr<IGSPHSetupNode> make_modifier_add_offset(
            SetupNodePtr parent, Tvec offset_postion, Tvec offset_velocity);

        std::shared_ptr<IGSPHSetupNode> make_modifier_filter(
            SetupNodePtr parent, std::function<bool(Tvec)> filter);

        std::shared_ptr<IGSPHSetupNode> make_modifier_split_part(
            SetupNodePtr parent, u64 n_split, u64 seed, Tscal h_scaling);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        u64 injected_parts = 0;
    };

} // namespace shammodels::gsph::modules

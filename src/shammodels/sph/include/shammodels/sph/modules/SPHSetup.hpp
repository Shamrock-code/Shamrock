// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SPHSetup.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <memory>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class SPHSetup {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SPHSetup(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void apply_setup(
            SetupNodePtr setup,
            bool part_reordering,
            std::optional<u32> insert_step = std::nullopt);

        void apply_setup_new(
            SetupNodePtr setup,
            bool part_reordering,
            std::optional<u32> insert_count_per_step            = std::nullopt,
            std::optional<u64> max_msg_count_per_rank_per_step  = std::nullopt,
            std::optional<u64> max_data_count_per_rank_per_step = std::nullopt,
            bool do_setup_log                                   = false);

        std::shared_ptr<ISPHSetupNode> make_generator_lattice_hcp(
            Tscal dr, std::pair<Tvec, Tvec> box);

        std::shared_ptr<ISPHSetupNode> make_generator_disc_mc(
            Tscal part_mass,
            Tscal disc_mass,
            Tscal r_in,
            Tscal r_out,
            std::function<Tscal(Tscal)> sigma_profile,
            std::function<Tscal(Tscal)> H_profile,
            std::function<Tscal(Tscal)> rot_profile,
            std::function<Tscal(Tscal)> cs_profile,
            std::mt19937 eng,
            Tscal init_h_factor);

        std::shared_ptr<ISPHSetupNode> make_combiner_add(
            SetupNodePtr parent1, SetupNodePtr parent2);

        std::shared_ptr<ISPHSetupNode> make_modifier_warp_disc(
            SetupNodePtr parent, Tscal Rwarp, Tscal Hwarp, Tscal inclination, Tscal posangle);

        std::shared_ptr<ISPHSetupNode> make_modifier_custom_warp(
            SetupNodePtr parent,
            std::function<Tscal(Tscal)> inc_profile,
            std::function<Tscal(Tscal)> psi_profile,
            std::function<Tvec(Tscal)> k_profile);

        std::shared_ptr<ISPHSetupNode> make_modifier_add_offset(
            SetupNodePtr parent, Tvec offset_postion, Tvec offset_velocity);

        std::shared_ptr<ISPHSetupNode> make_modifier_filter(
            SetupNodePtr parent, std::function<bool(Tvec)> filter);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        u64 injected_parts = 0;
    };

} // namespace shammodels::sph::modules

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
 * @file VTKDump.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief VTK dump module for GSPH solver
 *
 * Physics-agnostic VTK output. Reads pre-computed fields from storage.
 * Field selection is delegated to the physics mode via get_output_field_names().
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <string>
#include <vector>

namespace shammodels::gsph::modules {

    /**
     * @brief VTK dump module for GSPH solver
     *
     * Exports particle data to VTK format for visualization.
     * Reads pre-computed fields directly from storage (physics-agnostic).
     *
     * Field selection is NOT hardcoded - the caller provides a list of field
     * names (from physics_mode->get_output_field_names()) and this module
     * writes whatever fields it's given.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class VTKDump {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage; ///< Storage containing pre-computed fields

        VTKDump(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Dump VTK file with specified physics fields
         *
         * @param filename Output filename
         * @param add_patch_world_id Whether to add patch/world IDs
         * @param physics_field_names Field names from physics_mode->get_output_field_names()
         *
         * Fields are looked up by name in storage.scalar_fields and storage.vector_fields.
         * Physics modes register their fields in these maps during init_fields().
         * This allows fully physics-agnostic VTK output.
         */
        void do_dump(
            std::string filename,
            bool add_patch_world_id,
            const std::vector<std::string> &physics_field_names);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::gsph::modules

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
 * @file ParticleInjector.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Insert the particles of the configured injectors into the patches of a scheduler.
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/injection/FallbackStreamSchedule.hpp"
#include "shammodels/common/injection/ParticleInjectionConfig.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <vector>

namespace shammodels::common::injection {

    struct InjectionReport {
        u64 born_walls     = 0; // Walls that crossed their injection point during the step
        u64 born_parts     = 0; // Particles in those walls
        u64 boundary_parts = 0; // Boundary particles re-created at the end of the step
    };

    /**
     * @brief Runtime state of the particle injectors of a solver
     *
     * every rank generates the same particles and only inserts the ones falling in its own patches, so injecting requires no MPI communication. The injected particles are tagged in the part_id field (see ParticleInjectionConfig.hpp).
     */
    template<class Tvec>
    class ParticleInjector {
        public:
        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @param part_mass mass of the particles
         * @param hfact ratio of the smoothing length to the mean particle spacing
         */
        ParticleInjector(const ParticleInjectionConfig<Tvec> &config, Tscal part_mass, Tscal hfact);

        /**
         * @brief Inject the particles born during the step (t_old, t_new]
         * Removes the boundary particles of the previous step, then adds the walls born during the step at their position at t_new, and the boundary walls at t_new. The walls born at or before t_old on the first call are assumed to be already in the simulation (restart).
         */
        InjectionReport inject(PatchScheduler &sched, Tscal t_old, Tscal t_new);

        /// Largest timestep allowed by the injectors at time t (no step over an injection start)
        Tscal get_max_dt(Tscal t) const;

        private:
        /// Particles to inject, generated identically on every rank
        struct HostBatch {
            std::vector<Tvec> xyz;
            std::vector<Tvec> vxyz;
            std::vector<Tscal> hpart;
            std::vector<Tscal> uint;
            std::vector<u64> part_id;
        };

        // Append the particles of a wall at time t, with ids id_base | (first_index + i)
        void append_wall(
            HostBatch &batch,
            u32 injector_idx,
            const StreamWall<Tscal> &wall,
            Tscal t,
            u64 id_base,
            u64 first_index);

        // Remove the boundary particles from the local patches
        void remove_boundary_parts(PatchScheduler &sched);

        // Insert in each local patch the particles of the batch that it contains
        void insert_batch(PatchScheduler &sched, HostBatch &batch);

        std::vector<FallbackStream<Tvec>> streams;
        std::vector<FallbackStreamSchedule<Tvec>> schedules;
        bool is_first_inject = true;
    };

} // namespace shammodels::common::injection

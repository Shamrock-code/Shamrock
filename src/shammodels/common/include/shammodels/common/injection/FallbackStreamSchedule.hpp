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
 * @file FallbackStreamSchedule.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Deterministic sequence of particle walls of a fallback stream. Loosly inspired from
 * Phantom's firehose setup.
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/injection/ParticleInjectionConfig.hpp"
#include <array>
#include <deque>
#include <map>
#include <vector>

namespace shammodels::common::injection {

    template<class Tscal>
    struct StreamWall {
        Tscal t_birth;  // Time at which the wall crosses the injection point
        Tscal hpart;    // Initial smoothing length of the particles of the wall (constant because
                        // lattice)
        u32 lattice_n;  // Number of lattice sites along a diameter of the cross-section
        u32 part_count; // Number of particles in the wall
        u64 first_part; // Index of the first particle of the wall among those of the stream
    };

    /**
     * @brief Deterministic sequence of the walls of a FallbackStream
     *
     * Wall i is born (crosses the injection point) when the injected mass reaches the mass of the
     * walls 0 to i, so that the mean rate follows the fallback rate whatever the timestep. The
     * lattice of wall i is chosen from the rate at the birth of wall i-1 (t_start for wall 0).
     * The sequence only depends on the stream config, the particle mass and hfact: it is the same
     * on every MPI rank, and can be replayed from the start after a restart.
     */
    template<class Tvec>
    class FallbackStreamSchedule {
        public:
        using Tscal = shambase::VecComponent<Tvec>;
        using Wall  = StreamWall<Tscal>;
        /// Offset of a lattice site in the stream cross-section, along the basis (e1, e2)
        using Site = std::array<Tscal, 2>;

        /// Largest number of lattice sites along a diameter of the stream
        static constexpr u32 max_lattice_n = 4096;

        /**
         * @param part_mass mass of the injected particles
         * @param hfact ratio of the smoothing length to the mean particle spacing
         */
        FallbackStreamSchedule(const FallbackStream<Tvec> &stream, Tscal part_mass, Tscal hfact);

        /// Remove and return the walls born at or before t, in birth order
        std::vector<Wall> pop_born_walls(Tscal t);

        /// The next n walls not born yet (fewer if the stream stops before)
        std::vector<Wall> peek_unborn_walls(u32 n);

        /// Sites of the cross-section lattice having lattice_n sites along a diameter
        const std::vector<Site> &get_lattice(u32 lattice_n);

        /// Unit vectors (e1, e2) spanning the stream cross-section
        inline const std::array<Tvec, 2> &get_cross_section_basis() const { return basis; }

        /// Number of particles in the walls popped so far
        inline u64 get_born_part_count() const { return born_part_count; }

        private:
        /// Append the next wall of the sequence to unborn_walls, false if the stream has stopped
        bool generate_next_wall();

        FallbackStream<Tvec> stream;
        Tscal part_mass;
        Tscal hfact;
        std::array<Tvec, 2> basis;

        std::deque<Wall> unborn_walls; // Generated walls that are not born yet
        u64 generated_part_count = 0;  // Particles in all the generated walls
        u64 born_part_count      = 0;  // Particles in the popped walls
        Tscal t_last_birth;            // Birth time of the last generated wall
        bool stopped = false;          // The stream will not generate any more wall

        std::map<u32, std::vector<Site>> lattices;
    };

} // namespace shammodels::common::injection

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
 * @file SRIterateSmoothingLength.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SR-specific volume-based smoothing length iteration (Kitajima et al. 2025)
 *
 * This module implements the volume-based approach for computing smoothing length h
 * used in Special Relativistic GSPH.
 *
 * Unlike standard SPH where h is a Lagrangian particle property, this approach treats
 * h as a field quantity computed from local density:
 *
 * Key differences from standard SPH:
 * 1. Uses Kitajima C_smooth factor: W(r_ij, C_smooth * h_i)
 * 2. Computes h directly from volume: h = hfact * V^(1/dim)
 * 3. No Newton iteration needed - pure volume-based iteration
 *
 * This gives smooth monotonic h variation across discontinuities, matching
 * Kitajima, Inutsuka & Seno (2025) arXiv:2510.18251.
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

namespace shammodels::gsph::physics::sr {

    template<class Tvec, class SPHKernel>
    class SRIterateSmoothingLength : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal gpart_mass;
        Tscal h_evol_max;      ///< Max total h evolution per subcycle
        Tscal h_evol_iter_max; ///< Max h evolution per iteration
        Tscal c_smooth;        ///< Kitajima C_smooth factor

        public:
        SRIterateSmoothingLength(
            Tscal gpart_mass, Tscal h_evol_max, Tscal h_evol_iter_max, Tscal c_smooth = Tscal{1})
            : gpart_mass(gpart_mass), h_evol_max(h_evol_max), h_evol_iter_max(h_evol_iter_max),
              c_smooth(c_smooth) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &positions;
            const shamrock::solvergraph::IFieldSpan<Tscal> &old_h;
            shamrock::solvergraph::IFieldSpan<Tscal> &new_h;
            shamrock::solvergraph::IFieldSpan<Tscal> &eps_h;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> positions,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> old_h,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> new_h,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> eps_h) {
            __internal_set_ro_edges({sizes, neigh_cache, positions, old_h});
            __internal_set_rw_edges({new_h, eps_h});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "SRIterateSmoothingLength";
        };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::gsph::physics::sr

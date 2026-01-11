// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RiemannResultEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph edge for Riemann solver results
 *
 * Stores the interface state (p*, v*) from Riemann problem solutions.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Result from Riemann solver for a particle pair
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct RiemannResult {
        Tscal p_star; ///< Interface pressure
        Tscal v_star; ///< Interface velocity (along pair axis)
    };

    /**
     * @brief Edge holding Riemann solver results
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    class RiemannResultEdge : public shamrock::solvergraph::IEdgeNamed {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        using Result = RiemannResult<Tscal>;

        RiemannResultEdge(std::string name, std::string tex_symbol)
            : IEdgeNamed(std::move(name), std::move(tex_symbol)) {}

        void free_alloc() override {}
    };

} // namespace shammodels::gsph::solvergraph

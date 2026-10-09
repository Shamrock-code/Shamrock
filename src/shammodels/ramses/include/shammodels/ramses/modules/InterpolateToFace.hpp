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
 * @file InterpolateToFace.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <array>

// Note on the edge lists below: the Euler time derivatives (dt_rho, dt_vel,
// dt_press and their dust counterparts) used to be recomputed inside these
// kernels for both sides of every link. They are now precomputed per cell by
// NodeEulerTimeDerivativeGas / NodeEulerTimeDerivativeDust and merely loaded
// here, which is why each node only carries the fields its spatial
// reconstruction still needs.

#define NODE_EDGES_RHO(X_RO, X_RW)                                                                 \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(ScalarEdgeScal, dt_interp)                                                                \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhos)                                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_rho)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldScal, rho_face_xp)                                                               \
    X_RW(LinkFieldScal, rho_face_xm)                                                               \
    X_RW(LinkFieldScal, rho_face_yp)                                                               \
    X_RW(LinkFieldScal, rho_face_ym)                                                               \
    X_RW(LinkFieldScal, rho_face_zp)                                                               \
    X_RW(LinkFieldScal, rho_face_zm)

#define NODE_EDGES_VEL(X_RO, X_RW)                                                                 \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(ScalarEdgeScal, dt_interp)                                                                \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dx_vel)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dy_vel)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dz_vel)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_vel)                                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldVec, vel_face_xp)                                                                \
    X_RW(LinkFieldVec, vel_face_xm)                                                                \
    X_RW(LinkFieldVec, vel_face_yp)                                                                \
    X_RW(LinkFieldVec, vel_face_ym)                                                                \
    X_RW(LinkFieldVec, vel_face_zp)                                                                \
    X_RW(LinkFieldVec, vel_face_zm)

#define NODE_EDGES_PRESS(X_RO, X_RW)                                                               \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(ScalarEdgeScal, dt_interp)                                                                \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_press)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_P)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_press)                                 \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldScal, press_face_xp)                                                             \
    X_RW(LinkFieldScal, press_face_xm)                                                             \
    X_RW(LinkFieldScal, press_face_yp)                                                             \
    X_RW(LinkFieldScal, press_face_ym)                                                             \
    X_RW(LinkFieldScal, press_face_zp)                                                             \
    X_RW(LinkFieldScal, press_face_zm)

#define NODE_EDGES_RHO_DUST(X_RO, X_RW)                                                            \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(ScalarEdgeScal, dt_interp)                                                                \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhos_dust)                                \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_rho_dust)                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho_dust)                              \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldScal, rho_dust_face_xp)                                                          \
    X_RW(LinkFieldScal, rho_dust_face_xm)                                                          \
    X_RW(LinkFieldScal, rho_dust_face_yp)                                                          \
    X_RW(LinkFieldScal, rho_dust_face_ym)                                                          \
    X_RW(LinkFieldScal, rho_dust_face_zp)                                                          \
    X_RW(LinkFieldScal, rho_dust_face_zm)

#define NODE_EDGES_VEL_DUST(X_RO, X_RW)                                                            \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(ScalarEdgeScal, dt_interp)                                                                \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel_dust)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dx_vel_dust)                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dy_vel_dust)                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dz_vel_dust)                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_vel_dust)                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldVec, vel_dust_face_xp)                                                           \
    X_RW(LinkFieldVec, vel_dust_face_xm)                                                           \
    X_RW(LinkFieldVec, vel_dust_face_yp)                                                           \
    X_RW(LinkFieldVec, vel_dust_face_ym)                                                           \
    X_RW(LinkFieldVec, vel_dust_face_zp)                                                           \
    X_RW(LinkFieldVec, vel_dust_face_zm)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class InterpolateToFaceRho : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using ScalarEdgeScal = shamrock::solvergraph::ScalarEdge<Tscal>;
        using AMRGraphEdge   = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldScal  = solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>;

        u32 block_size;

        public:
        InterpolateToFaceRho(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES_RHO)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "InterpolateRhoToFaceRho"; };

        virtual std::string _impl_get_tex() const;
    };

    template<class Tvec, class TgridVec>
    class InterpolateToFaceVel : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using ScalarEdgeScal = shamrock::solvergraph::ScalarEdge<Tscal>;
        using AMRGraphEdge   = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldVec   = solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>;

        u32 block_size;

        public:
        InterpolateToFaceVel(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES_VEL)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "InterpolateVelToFaceVel"; };

        virtual std::string _impl_get_tex() const;
    };

    template<class Tvec, class TgridVec>
    class InterpolateToFacePress : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using ScalarEdgeScal = shamrock::solvergraph::ScalarEdge<Tscal>;
        using AMRGraphEdge   = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldScal  = solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>;

        u32 block_size;

        public:
        InterpolateToFacePress(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES_PRESS)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "InterpolatePressToFacePress";
        };

        virtual std::string _impl_get_tex() const;
    };

    template<class Tvec, class TgridVec>
    class InterpolateToFaceRhoDust : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using ScalarEdgeScal = shamrock::solvergraph::ScalarEdge<Tscal>;
        using AMRGraphEdge   = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldScal  = solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>;

        u32 block_size;
        u32 ndust;

        public:
        InterpolateToFaceRhoDust(u32 block_size, u32 ndust)
            : block_size(block_size), ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES_RHO_DUST)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "InterpolateRhoDustToFaceRhoDust";
        };

        virtual std::string _impl_get_tex() const;
    };

    template<class Tvec, class TgridVec>
    class InterpolateToFaceVelDust : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using ScalarEdgeScal = shamrock::solvergraph::ScalarEdge<Tscal>;
        using AMRGraphEdge   = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldVec   = solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>;

        u32 block_size;
        u32 ndust;

        public:
        InterpolateToFaceVelDust(u32 block_size, u32 ndust)
            : block_size(block_size), ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES_VEL_DUST)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "InterpolateVelDustToFaceVelDust";
        };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES_RHO
#undef NODE_EDGES_VEL
#undef NODE_EDGES_PRESS
#undef NODE_EDGES_RHO_DUST
#undef NODE_EDGES_VEL_DUST

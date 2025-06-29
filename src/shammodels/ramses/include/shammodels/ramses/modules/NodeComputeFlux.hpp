// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeComputeFlux.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/NeighGrapkLinkFieldEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"

namespace shammodels::basegodunov::modules {

    using RiemannSolverMode     = shammodels::basegodunov::RiemmanSolverMode;
    using DustRiemannSolverMode = shammodels::basegodunov::DustRiemannSolverMode;
    using Direction             = shammodels::basegodunov::modules::Direction;

    template<class Tvec, class TgridVec, RiemannSolverMode mode, Direction dir>
    class NodeComputeFluxGasDirMode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal gamma;

        public:
        NodeComputeFluxGasDirMode(Tscal gamma) : gamma(gamma) {}

        struct Edges {
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rho_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tvec> &flux_rhov_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rhoe_face;
        };

        inline void set_edges(
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> press_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face) {
            __internal_set_ro_edges({cell_neigh_graph, rho_face, vel_face, press_face});
            __internal_set_rw_edges({flux_rho_face, flux_rhov_face, flux_rhoe_face});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(0),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(1),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(2),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(3),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(0),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(1),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(2),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeFluxGasDirMode"; };

        virtual std::string _impl_get_tex();
    };

    template<class Tvec, class TgridVec, DustRiemannSolverMode mode, Direction dir>
    class NodeComputeFluxDustDirMode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        NodeComputeFluxDustDirMode(u32 ndust) : ndust(ndust) {}

        struct Edges {
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rho_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tvec> &flux_rhov_face;
        };

        inline void set_edges(
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face) {
            __internal_set_ro_edges({cell_neigh_graph, rho_face, vel_face});
            __internal_set_rw_edges({flux_rho_face, flux_rhov_face});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(0),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(1),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(2),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(0),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeFluxDustDirMode"; };

        virtual std::string _impl_get_tex();
    };

    template<class Tvec, class TgridVec, RiemannSolverMode mode>
    class NodeComputeFluxGasMode : public shamrock::solvergraph::OperationSequence {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static auto make_sequence(
            Tscal gamma,

            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_zm

            ) -> std::vector<std::shared_ptr<shamrock::solvergraph::INode>> {

            std::vector<std::shared_ptr<shamrock::solvergraph::INode>> flux_sequence;

            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::xm>
                node_xm(gamma);
            node_xm.set_edges(
                cell_neigh_graph,
                rho_face_xm,
                vel_face_xm,
                press_face_xm,
                flux_rho_face_xm,
                flux_rhov_face_xm,
                flux_rhoe_face_xm);
            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::xp>
                node_xp(gamma);
            node_xp.set_edges(
                cell_neigh_graph,
                rho_face_xp,
                vel_face_xp,
                press_face_xp,
                flux_rho_face_xp,
                flux_rhov_face_xp,
                flux_rhoe_face_xp);

            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::ym>
                node_ym(gamma);
            node_ym.set_edges(
                cell_neigh_graph,
                rho_face_ym,
                vel_face_ym,
                press_face_ym,
                flux_rho_face_ym,
                flux_rhov_face_ym,
                flux_rhoe_face_ym);
            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::yp>
                node_yp(gamma);
            node_yp.set_edges(
                cell_neigh_graph,
                rho_face_yp,
                vel_face_yp,
                press_face_yp,
                flux_rho_face_yp,
                flux_rhov_face_yp,
                flux_rhoe_face_yp);
            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::zm>
                node_zm(gamma);
            node_zm.set_edges(
                cell_neigh_graph,
                rho_face_zm,
                vel_face_zm,
                press_face_zm,
                flux_rho_face_zm,
                flux_rhov_face_zm,
                flux_rhoe_face_zm);
            modules::NodeComputeFluxGasDirMode<Tvec, TgridVec, mode, modules::Direction::zp>
                node_zp(gamma);
            node_zp.set_edges(
                cell_neigh_graph,
                rho_face_zp,
                vel_face_zp,
                press_face_zp,
                flux_rho_face_zp,
                flux_rhov_face_zp,
                flux_rhoe_face_zp);

            flux_sequence.push_back(std::make_shared<decltype(node_xm)>(std::move(node_xm)));
            flux_sequence.push_back(std::make_shared<decltype(node_xp)>(std::move(node_xp)));
            flux_sequence.push_back(std::make_shared<decltype(node_ym)>(std::move(node_ym)));
            flux_sequence.push_back(std::make_shared<decltype(node_yp)>(std::move(node_yp)));
            flux_sequence.push_back(std::make_shared<decltype(node_zm)>(std::move(node_zm)));
            flux_sequence.push_back(std::make_shared<decltype(node_zp)>(std::move(node_zp)));

            return flux_sequence;
        };

        public:
        NodeComputeFluxGasMode(
            std::string name,
            Tscal gamma,

            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                press_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face_zm)
            : OperationSequence(
                  std::move(name),
                  make_sequence(
                      gamma,
                      cell_neigh_graph,
                      rho_face_xp,
                      rho_face_xm,
                      rho_face_yp,
                      rho_face_ym,
                      rho_face_zp,
                      rho_face_zm,
                      vel_face_xp,
                      vel_face_xm,
                      vel_face_yp,
                      vel_face_ym,
                      vel_face_zp,
                      vel_face_zm,
                      press_face_xp,
                      press_face_xm,
                      press_face_yp,
                      press_face_ym,
                      press_face_zp,
                      press_face_zm,
                      flux_rho_face_xp,
                      flux_rho_face_xm,
                      flux_rho_face_yp,
                      flux_rho_face_ym,
                      flux_rho_face_zp,
                      flux_rho_face_zm,
                      flux_rhov_face_xp,
                      flux_rhov_face_xm,
                      flux_rhov_face_yp,
                      flux_rhov_face_ym,
                      flux_rhov_face_zp,
                      flux_rhov_face_zm,
                      flux_rhoe_face_xp,
                      flux_rhoe_face_xm,
                      flux_rhoe_face_yp,
                      flux_rhoe_face_ym,
                      flux_rhoe_face_zp,
                      flux_rhoe_face_zm)) {}
    };

    template<class Tvec, class TgridVec, DustRiemannSolverMode mode>
    class NodeComputeFluxDustMode : public shamrock::solvergraph::OperationSequence {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static auto make_sequence(
            u32 ndust,

            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zm

            ) -> std::vector<std::shared_ptr<shamrock::solvergraph::INode>> {

            std::vector<std::shared_ptr<shamrock::solvergraph::INode>> flux_sequence;

            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::xm>
                node_xm(ndust);
            node_xm.set_edges(
                cell_neigh_graph, rho_face_xm, vel_face_xm, flux_rho_face_xm, flux_rhov_face_xm);
            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::xp>
                node_xp(ndust);
            node_xp.set_edges(
                cell_neigh_graph, rho_face_xp, vel_face_xp, flux_rho_face_xp, flux_rhov_face_xp);

            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::ym>
                node_ym(ndust);
            node_ym.set_edges(
                cell_neigh_graph, rho_face_ym, vel_face_ym, flux_rho_face_ym, flux_rhov_face_ym);
            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::yp>
                node_yp(ndust);
            node_yp.set_edges(
                cell_neigh_graph, rho_face_yp, vel_face_yp, flux_rho_face_yp, flux_rhov_face_yp);
            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::zm>
                node_zm(ndust);
            node_zm.set_edges(
                cell_neigh_graph, rho_face_zm, vel_face_zm, flux_rho_face_zm, flux_rhov_face_zm);
            modules::NodeComputeFluxDustDirMode<Tvec, TgridVec, mode, modules::Direction::zp>
                node_zp(ndust);
            node_zp.set_edges(
                cell_neigh_graph, rho_face_zp, vel_face_zp, flux_rho_face_zp, flux_rhov_face_zp);

            flux_sequence.push_back(std::make_shared<decltype(node_xm)>(std::move(node_xm)));
            flux_sequence.push_back(std::make_shared<decltype(node_xp)>(std::move(node_xp)));
            flux_sequence.push_back(std::make_shared<decltype(node_ym)>(std::move(node_ym)));
            flux_sequence.push_back(std::make_shared<decltype(node_yp)>(std::move(node_yp)));
            flux_sequence.push_back(std::make_shared<decltype(node_zm)>(std::move(node_zm)));
            flux_sequence.push_back(std::make_shared<decltype(node_zp)>(std::move(node_zp)));

            return flux_sequence;
        };

        public:
        NodeComputeFluxDustMode(
            std::string name,
            u32 ndust,

            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face_zm,

            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face_zm)
            : OperationSequence(
                  std::move(name),
                  make_sequence(
                      ndust,
                      cell_neigh_graph,
                      rho_face_xp,
                      rho_face_xm,
                      rho_face_yp,
                      rho_face_ym,
                      rho_face_zp,
                      rho_face_zm,
                      vel_face_xp,
                      vel_face_xm,
                      vel_face_yp,
                      vel_face_ym,
                      vel_face_zp,
                      vel_face_zm,
                      flux_rho_face_xp,
                      flux_rho_face_xm,
                      flux_rho_face_yp,
                      flux_rho_face_ym,
                      flux_rho_face_zp,
                      flux_rho_face_zm,
                      flux_rhov_face_xp,
                      flux_rhov_face_xm,
                      flux_rhov_face_yp,
                      flux_rhov_face_ym,
                      flux_rhov_face_zp,
                      flux_rhov_face_zm)) {}
    };

} // namespace shammodels::basegodunov::modules

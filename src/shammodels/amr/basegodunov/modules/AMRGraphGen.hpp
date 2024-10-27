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
 * @file AMRGraphGen.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/amr/NeighGraph.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class AMRGraphGen {

        class AMRBlockFinder;
        class AMRLowering;

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton         = u64;
        using AMRBlock         = typename Config::AMRBlock;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        AMRGraphGen(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Return the AMR graph for the block links (two blocks having a common face).
         * Could be a morton code lookup in the tree.
         * @return shambase::DistributedData<OrientedAMRGraph>
         */
        shambase::DistributedData<OrientedAMRGraph> find_AMR_block_graph_links_common_face();

        /**
         * @brief lower the AMR block graph to a cell graph
         *
         * @param oriented_block_graph
         */
        void lower_AMR_block_graph_to_cell_stencil_graph(
            shambase::DistributedData<OrientedAMRGraph> &block_graph_links);
        /**
         * @brief lower the AMR block graph to a cell graph
         *
         * @param oriented_block_graph
         */
        void lower_AMR_block_graph_to_cell_common_face_graph(
            shambase::DistributedData<OrientedAMRGraph> &block_graph_links);
    };

} // namespace shammodels::basegodunov::modules

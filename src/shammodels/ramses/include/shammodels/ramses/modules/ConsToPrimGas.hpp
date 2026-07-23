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
 * @file ConsToPrimGas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shammodels/ramses//SolverConfig.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)                                     \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_P)

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    class NodeConsToPrimGas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        Tscal gamma;
        using Config = SolverConfig<Tvec, TgridVec>;
        Config::EOSConfig &eos_config;

        public:
        NodeConsToPrimGas(u32 block_size, Tscal gamma, Config::EOSConfig &eos_config) : block_size(block_size), gamma(gamma), eos_config(eos_config) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ConsToPrimGas"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES

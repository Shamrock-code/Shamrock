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
 * @file LoopSmoothingLengthIter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the LoopSmoothingLengthIter module for looping over the smoothing length
 * iteration until convergence.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, eps_h)                                          \
                                                                                                   \
    X_RW(shamrock::solvergraph::IDataEdge<bool>, is_converged)

namespace shammodels::sph::modules {

    template<class Tvec>
    class LoopSmoothingLengthIter : public shamrock::solvergraph::INode {

        std::shared_ptr<INode> iterate_smth_length_once_ptr;

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal epsilon_h;
        u32 h_iter_per_subcycles;
        bool print_info;

        public:
        LoopSmoothingLengthIter(
            std::shared_ptr<INode> iterate_smth_length_once_ptr,
            Tscal epsilon_h,
            u32 h_iter_per_subcycles,
            bool print_info)
            : iterate_smth_length_once_ptr(std::move(iterate_smth_length_once_ptr)),
              epsilon_h(epsilon_h), h_iter_per_subcycles(h_iter_per_subcycles),
              print_info(print_info) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "LoopSmoothingLengthIter"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES

template class shammodels::sph::modules::LoopSmoothingLengthIter<f64_3>;

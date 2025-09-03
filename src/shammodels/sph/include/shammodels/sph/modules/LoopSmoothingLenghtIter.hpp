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
 * @file LoopSmoothingLenghtIter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the IterateSmoothingLengthDensity module for iterating smoothing length based on
 * the SPH density sum.
 */

#include "shambase/memory.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>

namespace shammodels::sph::modules {

    template<class Tvec>
    class LoopSmoothingLenghtIter : public shamrock::solvergraph::INode {

        std::shared_ptr<INode> iterate_smth_lenght_once_ptr;

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal epsilon_h;
        u32 h_iter_per_subcycles;

        public:
        LoopSmoothingLenghtIter(
            std::shared_ptr<INode> iterate_smth_lenght_once_ptr,
            Tscal epsilon_h,
            u32 h_iter_per_subcycles)
            : iterate_smth_lenght_once_ptr(iterate_smth_lenght_once_ptr), epsilon_h(epsilon_h),
              h_iter_per_subcycles(h_iter_per_subcycles) {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<Tscal> &eps_h;
            shamrock::solvergraph::ScalarEdge<bool> &is_converged;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> eps_h,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_converged) {
            __internal_set_ro_edges({eps_h});
            __internal_set_rw_edges({is_converged});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<bool>>(0),
            };
        }

        void _impl_evaluate_internal() {
            StackEntry stack_loc{};

            auto edges = get_edges();

            auto &eps_h        = edges.eps_h;
            auto &is_converged = edges.is_converged;

            Tscal local_min_eps_h = -1;
            Tscal local_max_eps_h = -1;

            for (u32 iter_h = 0; iter_h < h_iter_per_subcycles; iter_h++) {

                shambase::get_check_ref(iterate_smth_lenght_once_ptr).evaluate();

                local_max_eps_h = shamrock::solvergraph::get_rank_max(eps_h);

                shamcomm::logs::raw_ln(
                    shamrock::solvergraph::get_rank_min(eps_h),
                    shamrock::solvergraph::get_rank_max(eps_h));

                shamlog_debug_ln(
                    "Smoothinglength", "iteration :", iter_h, "epsmax", local_max_eps_h);

                // either converged or require gz re-exchange
                if (local_max_eps_h < epsilon_h) {
                    break;
                }
            }

            local_min_eps_h = shamrock::solvergraph::get_rank_min(eps_h);

            // if a particle need a gz update eps_h is set to -1
            bool local_should_rerun_gz = local_min_eps_h < 0;
            bool local_is_h_below_tol  = local_max_eps_h < epsilon_h;

            bool local_is_converged = local_is_h_below_tol && (!local_should_rerun_gz);

            is_converged.value
                = shamalgs::collective::are_all_rank_true(local_is_converged, MPI_COMM_WORLD);
        }

        inline virtual std::string _impl_get_label() { return "LoopSmoothingLenghtIter"; };

        inline virtual std::string _impl_get_tex() { return "todo"; }
    };
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::LoopSmoothingLenghtIter<f64_3>;

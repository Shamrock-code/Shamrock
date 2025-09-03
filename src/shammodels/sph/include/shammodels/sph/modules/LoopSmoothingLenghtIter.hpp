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
#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
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
            shamrock::solvergraph::ScalarEdge<bool> &is_h_below_tol;
            shamrock::solvergraph::ScalarEdge<bool> &should_rerun_gz;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> eps_h,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_h_below_tol,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> should_rerun_gz) {
            __internal_set_ro_edges({eps_h});
            __internal_set_rw_edges({is_h_below_tol, should_rerun_gz});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<bool>>(0),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<bool>>(1)};
        }

        void _impl_evaluate_internal() {
            StackEntry stack_loc{};

            auto edges = get_edges();

            auto &eps_h           = edges.eps_h.get_spans();
            auto &is_h_below_tol  = edges.is_h_below_tol;
            auto &should_rerun_gz = edges.should_rerun_gz;

            auto get_local_eps_max = [&]() {
                Tscal local_max_eps_h = shambase::VectorProperties<Tscal>::get_min();
                eps_h.get_refs().for_each([&](u64 id, Tscal &eps) {
                    local_max_eps_h = sham::max(local_max_eps_h, eps);
                });
                return local_max_eps_h;
            };
            Tscal local_max_eps_h = eps_h.compute_rank_max();
            return shamalgs::collective::allreduce_max(local_max_eps_h);
        };

        u32 iter_h = 0;
        for (; iter_h < h_iter_per_subcycles; iter_h++) {

            shambase::get_check_ref(iterate_smth_lenght_once_ptr).evaluate();

            Tscal max_eps_h = eps_h.compute_rank_max();

            shamlog_debug_ln("Smoothinglength", "iteration :", iter_h, "epsmax", max_eps_h);

            // either converged or require gz re-exchange
            if (max_eps_h < epsilon_h) {
                break;
            }
        }

        Tscal local_min_eps_h = eps_h.compute_rank_min();
        Tscal local_max_eps_h = eps_h.compute_rank_max();

        Tscal min_eps_h = shamalgs::collective::allreduce_min(local_min_eps_h);
        Tscal max_eps_h = shamalgs::collective::allreduce_max(local_max_eps_h);

        should_rerun_gz.value = min_eps_h == -1;
        is_h_below_tol.value  = max_eps_h < epsilon_h;
    }

    inline virtual std::string
    _impl_get_label() {
        return "LoopSmoothingLenghtIter";
    };

    virtual std::string _impl_get_tex();
};
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::LoopSmoothingLenghtIter<f64_3>;

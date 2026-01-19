// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file LoopSmoothingLengthIter.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the LoopSmoothingLengthIter module, which iterates smoothing length
 * based on the SPH density sum until convergence.
 */

#include "shambase/memory.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    void LoopSmoothingLengthIter<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        auto &eps_h        = edges.eps_h;
        auto &is_converged = edges.is_converged;

        Tscal local_min_eps_h = -1;
        Tscal local_max_eps_h = shambase::get_max<Tscal>();

        u32 iter_h = 0;
        for (; iter_h < h_iter_per_subcycles; iter_h++) {

            shambase::get_check_ref(iterate_smth_length_once_ptr).evaluate();

            local_max_eps_h = shamrock::solvergraph::get_rank_max(eps_h);

            shamlog_debug_ln("Smoothinglength", "iteration :", iter_h, "epsmax", local_max_eps_h);

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

        if (is_converged.value && print_info) {

            Tscal min_eps_h = shamalgs::collective::allreduce_min(local_min_eps_h);
            Tscal max_eps_h = shamalgs::collective::allreduce_max(local_max_eps_h);

            if (shamcomm::world_rank() == 0) {
                std::string log = "";
                log += "smoothing length iteration converged\n";
                log += shambase::format(
                    "  eps min = {}, max = {}\n  iterations = {}", min_eps_h, max_eps_h, iter_h);

                logger::info_ln("Smoothinglength", log);
            }
        }
    }

    template<class Tvec>
    std::string LoopSmoothingLengthIter<Tvec>::_impl_get_tex() const {
        return "TODO";
    }

} // namespace shammodels::sph::modules

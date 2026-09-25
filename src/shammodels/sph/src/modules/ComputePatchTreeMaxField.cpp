// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputePatchTreeMaxField.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/math.hpp"
#include "shammodels/sph/modules/ComputePatchTreeMaxField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/NodeInstance.hpp"

template<class Tvec>
void shammodels::sph::modules::ComputePatchTreeMaxField<Tvec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    auto edges = get_edges();

    edges.patchtree_max.patchtree_field = edges.patch_tree.get_patch_tree().make_patch_tree_field(
        shamsys::instance::get_compute_queue(),
        edges.patch_values.values,
        [](Tscal h0, Tscal h1, Tscal h2, Tscal h3, Tscal h4, Tscal h5, Tscal h6, Tscal h7) {
            return sham::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
        });
}

template<class Tvec>
std::string shammodels::sph::modules::ComputePatchTreeMaxField<Tvec>::_impl_get_tex() const {
    return "TODO";
}

template class shammodels::sph::modules::ComputePatchTreeMaxField<f64_3>;

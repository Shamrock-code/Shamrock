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
 * @file SetWhenMask.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Solver graph node overwriting a field with a constant value where a mask is set.
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<u32>, mask)                                             \
    X_RW(shamrock::solvergraph::IFieldSpan<T>, field_to_set)

namespace shammodels::sph::modules {

    /**
     * @brief Overwrite every variable of a field with a constant where the per-object mask is
     * non-zero
     *
     * The mask holds one value per object, the field holds `nvar` values per object. All the
     * `nvar` values of an object with a non-zero mask are set to `val_to_set`, the others are
     * left untouched.
     *
     * @tparam T type of the field values
     */
    template<class T>
    class SetWhenMask : public shamrock::solvergraph::INode {

        T val_to_set;
        u32 nvar;

        public:
        SetWhenMask(T val_to_set, u32 nvar = 1) : val_to_set(val_to_set), nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            __shamrock_stack_entry();

            auto edges = get_edges();

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            edges.mask.check_sizes(edges.part_counts.indexes);
            edges.field_to_set.ensure_sizes(edges.part_counts.indexes);

            auto var_counts = edges.part_counts.indexes.template map<u32>([&](u64, u32 count) {
                return count * nvar;
            });

            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{edges.mask.get_spans()},
                sham::DDMultiRef{edges.field_to_set.get_spans()},
                var_counts,
                [val_to_set = this->val_to_set,
                 nvar       = this->nvar](u32 id_var, const u32 *mask, T *field_to_set) {
                    if (mask[id_var / nvar] != 0) {
                        field_to_set[id_var] = val_to_set;
                    }
                });
        }

        inline virtual std::string _impl_get_label() const { return "SetWhenMask"; };

        inline virtual std::string _impl_get_tex() const {
            auto mask         = get_ro_edge_base(1).get_tex_symbol();
            auto field_to_set = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
    Set the field to a constant value \(c\) where the mask is non-zero
    \begin{align}
    {field}_{i,k} &= c \quad \text{if } {mask}_i \neq 0, \quad k = 0, \dots, {nvar} - 1
    \end{align}
    )tex";

            shambase::replace_all(tex, "{field}", field_to_set);
            shambase::replace_all(tex, "{mask}", mask);
            shambase::replace_all(tex, "{nvar}", std::to_string(nvar));

            return tex;
        }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

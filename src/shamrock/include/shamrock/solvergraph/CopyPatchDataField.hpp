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
 * @file CopyPatchDataField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the CopyPatchDataField class for copying fields between patch data field
 * references.
 *
 */

#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, original)                                           \
    X_RW(shamrock::solvergraph::IFieldRefs<T>, target)

namespace shamrock::solvergraph {

    /**
     * @brief A solver graph node that copies field data from source field references to target
     * fields.
     *
     * This node performs a deep copy of field data from a source IFieldRefs container to a target
     * Field. It handles distributed field data across multiple patches, ensuring that the target
     * field is properly sized and populated with data from the source.
     *
     * @tparam T The primitive type of the field data (e.g., f32, f64, u32, u64, f64_3)
     *
     * @code{.cpp}
     * // Create source and target field references
     * auto source_refs = FieldRefs<f32>::make_shared("source", "src");
     * auto target_field = std::make_shared<Field<f32>>(1, "target", "tgt");
     *
     * // Create and configure the copy node
     * auto copy_node = std::make_shared<CopyPatchDataField<f32>>();
     * copy_node->set_edges(source_refs, target_field);
     *
     * // Execute the copy operation
     * copy_node->evaluate();
     * @endcode
     *
     * The node will:
     * - Ensure the target field has the correct size for each patch
     * - Copy all field data from source to target maintaining data integrity
     * - Handle multiple patches automatically
     *
     * @author Timothée David--Cléris (tim.shamrock@proton.me)
     */
    template<class T>
    class CopyPatchDataField : public INode {

        public:
        /// Default constructor.
        CopyPatchDataField() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        /**
         * @brief Internal implementation of the field copying operation.
         *
         * This method performs the actual copy operation by:
         * 1. Collecting size information from the source fields
         * 2. Ensuring the target field has the correct size for each patch
         * 3. Copying the field data from source to target using overwrite operations
         */
        void _impl_evaluate_internal();

        /**
         * @brief Returns the human-readable label for this node.
         *
         * @return String identifier "CopyPatchDataField"
         */
        std::string _impl_get_label() const { return "CopyPatchDataField"; }

        /**
         * @brief Returns the LaTeX representation of this node for documentation.
         *
         * @return LaTeX string describing the copy operation with field symbols
         */
        std::string _impl_get_tex() const;
    };
} // namespace shamrock::solvergraph

#undef NODE_EDGES

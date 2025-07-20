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
 * @file IFieldSpan.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"

namespace shamrock::solvergraph {

    /// Alias for a DistributedData of PatchDataFieldSpans
    template<class T>
    using DDPatchDataFieldSpanPointer
        = shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>>;

    /**
     * @brief Interface for a solver graph edge representing a field as spans.
     *
     * Here a field refer to a field that is distributed over several patches.
     *
     * @tparam T The primitive type of the field
     */
    template<class T>
    class IFieldSpan : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        /// Get the DistributedData of spans attached to the underlying field
        virtual DDPatchDataFieldSpanPointer<T> &get_spans() = 0;

        /// Const variant of get_spans
        virtual const DDPatchDataFieldSpanPointer<T> &get_spans() const = 0;

        /**
         * @brief Check that the sizes of the patches in the field match the given
         * sizes.
         *
         * @param sizes the expected sizes
         */
        virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const = 0;

        /**
         * @brief Ensure that the sizes of the patches in the field match the given
         * sizes (Can resize the underlying fields).
         *
         * @param sizes the expected sizes
         */
        virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) = 0;
    };

} // namespace shamrock::solvergraph

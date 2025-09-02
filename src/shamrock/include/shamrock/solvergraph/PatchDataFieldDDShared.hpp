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
 * @file PatchDataFieldDDShared.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shared distributed data field for patch data management in solver graphs
 *
 * This header defines the PatchDataFieldDDShared template class, which provides a shared
 * distributed data interface for patch data fields within solver graph structures.
 * It extends IDataEdgeNamed to provide named data edge functionality while
 * managing shared patch data fields across distributed systems.
 *
 * The class is designed to work with the solver graph architecture, allowing
 * efficient sharing and management of typed patch data fields across multiple computational
 * nodes or processes.
 */

#include "shambase/DistributedDataShared.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief Shared distributed data field for patch data management
     *
     * PatchDataFieldDDShared provides a templated shared distributed data interface for
     * managing patch data fields within solver graph structures. It extends
     * IDataEdgeNamed to provide named data edge functionality while maintaining
     * shared patch data fields across distributed systems.
     *
     * This template class is particularly useful in scenarios where multiple solver
     * components need access to the same typed patch data field, ensuring efficient
     * memory usage and data consistency across distributed computations.
     *
     * @tparam T The data type stored in the patch data fields
     *
     * @code{.cpp}
     * // Create a shared distributed data field for double values
     * auto sharedField = std::make_unique<PatchDataFieldDDShared<double>>("label","tex symbol");
     *
     * // Access the shared patch data fields
     * auto& patchFields = sharedField->patchdata_fields;
     *
     * // Free allocated resources when done
     * sharedField->free_alloc();
     * @endcode
     */
    template<class T>
    class PatchDataFieldDDShared : public IEdgeNamed {

        public:
        /**
         * @brief Shared distributed data containing patch data fields
         *
         * This member provides access to shared distributed data that contains
         * typed patch data fields. The data is shared across multiple processes or
         * computational nodes, allowing efficient access to field information
         * without unnecessary data duplication.
         */
        shambase::DistributedDataShared<PatchDataField<T>> patchdata_fields;

        /// Inherit constructors from IDataEdgeNamed
        using IEdgeNamed::IEdgeNamed;

        /// Free allocated resources
        inline virtual void free_alloc() { patchdata_fields = {}; }
    };

} // namespace shamrock::solvergraph

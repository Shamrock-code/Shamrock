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
 * @file IFieldRefs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/math.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

namespace shamrock::solvergraph {

    /// Alias for a reference to a PatchDataField
    template<class T>
    using PatchDataFieldRef = std::reference_wrapper<PatchDataField<T>>;

    /// Alias for a DistributedData of PatchDataFieldRefs
    template<class T>
    using DDPatchDataFieldRef = shambase::DistributedData<PatchDataFieldRef<T>>;

    /**
     * @brief Interface for a solver graph edge representing a field as references to the underlying
     * patch fields.
     *
     * A field refer to a field that is distributed over several patches.
     *
     * @tparam T The primitive type of the field
     */
    template<class T>
    class IFieldRefs : public IFieldSpan<T> {
        public:
        using IFieldSpan<T>::IFieldSpan;

        /// Get the DistributedData of PatchDataFieldRefs
        virtual DDPatchDataFieldRef<T> &get_refs() = 0;

        /// Const variant of get_refs
        virtual const DDPatchDataFieldRef<T> &get_refs() const = 0;

        /// Get the underlying PatchDataField at the given id
        inline PatchDataField<T> &get_field(u64 id) const { return get_refs().get(id).get(); }
    };

    template<class T>
    T get_rank_max(const IFieldRefs<T> &field_refs) {
        T ret = shambase::VectorProperties<T>::get_min();
        field_refs.get_refs().for_each([&](u64 id, const PatchDataFieldRef<T> &field_ref) {
            ret = sham::max(ret, field_ref.get().compute_max());
        });
        return ret;
    }

    template<class T>
    T get_rank_min(const IFieldRefs<T> &field_refs) {
        T ret = shambase::VectorProperties<T>::get_max();
        field_refs.get_refs().for_each([&](u64 id, const PatchDataFieldRef<T> &field_ref) {
            ret = sham::min(ret, field_ref.get().compute_min());
        });
        return ret;
    }

} // namespace shamrock::solvergraph

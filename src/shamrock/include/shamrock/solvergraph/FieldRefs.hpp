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
 * @file FieldRefs.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include <functional>

namespace shamrock::solvergraph {

    template<class T>
    class FieldRefs : public IFieldRefs<T> {

        DDPatchDataFieldRef<T> field_refs;

        DDPatchDataFieldSpanPointer<T> spans;

        void sync_spans() {
            this->spans = field_refs.template map<shamrock::PatchDataFieldSpanPointer<T>>(
                [&](u64 id, std::reference_wrapper<PatchDataField<T>> &pdf) {
                    return pdf.get().get_pointer_span();
                });
        }

        public:
        using IFieldRefs<T>::IFieldRefs;

        virtual DDPatchDataFieldRef<T> &get_refs() { return field_refs; }

        virtual const DDPatchDataFieldRef<T> &get_refs() const { return field_refs; }

        virtual DDPatchDataFieldSpanPointer<T> &get_spans() { return spans; }

        virtual const DDPatchDataFieldSpanPointer<T> &get_spans() const { return spans; }

        inline virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const {
            on_distributeddata_diff(
                field_refs,
                sizes,
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing field ref in distributed data at id " + std::to_string(id));
                },
                [](u64 id) {
                    // TODO
                },
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra field ref in distributed data at id " + std::to_string(id));
                });
        }

        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {
            check_sizes(sizes);
        }

        void set_refs(DDPatchDataFieldRef<T> refs) {
            field_refs = refs;
            sync_spans();
        }

        DDPatchDataFieldRef<T> extract() {
            DDPatchDataFieldRef<T> refs = std::exchange(field_refs, {});
            sync_spans();
            return refs;
        }

        inline virtual PatchDataField<T> &get(u64 id_patch) const {
            return field_refs.get(id_patch);
        }

        inline virtual void free_alloc() { set_refs({}); }
    };
} // namespace shamrock::solvergraph

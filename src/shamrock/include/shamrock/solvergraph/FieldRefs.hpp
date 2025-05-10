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
#include <functional>

namespace shamrock::solvergraph {

    template<class T>
    class FieldRefs : public FieldSpan<T> {

        shambase::DistributedData<std::reference_wrapper<PatchDataField<T>>> field_refs;

        public:
        void
        set_refs(shambase::DistributedData<std::reference_wrapper<PatchDataField<T>>> field_refs) {
            this->field_refs = field_refs;
            this->spans      = field_refs.template map<shamrock::PatchDataFieldSpanPointer<T>>(
                [&](u64 id, std::reference_wrapper<PatchDataField<T>> &pdf) {
                    return pdf.get().get_pointer_span();
                });
        }

        auto extract_refs() {
            this->spans = {};
            return std::move(this->field_refs);
        }

        using FieldSpan<T>::FieldSpan;

        inline virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const {
            on_distributeddata_diff(
                field_refs,
                sizes,
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing field ref in distributed data at id " + std::to_string(id));
                },
                [](u64 id) {},
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra field ref in distributed data at id " + std::to_string(id));
                });
        }

        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {
            check_sizes(sizes);
        }
    };
} // namespace shamrock::solvergraph

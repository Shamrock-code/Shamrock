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
 * @file FieldSpan.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class FieldSpan : public IFieldSpan<T> {

        DDPatchDataFieldSpanPointer<T> spans;

        public:
        using IFieldSpan<T>::IFieldSpan;

        virtual DDPatchDataFieldSpanPointer<T> &get_spans() { return spans; }

        virtual const DDPatchDataFieldSpanPointer<T> &get_spans() const { return spans; }

        inline virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const {
            on_distributeddata_diff(
                spans,
                sizes,
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing field span in distributed data at id " + std::to_string(id));
                },
                [](u64 id) {
                    // TODO
                },
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra field span in distributed data at id " + std::to_string(id));
                });
        }

        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {
            check_sizes(sizes);
        }

        void set_spans(DDPatchDataFieldSpanPointer<T> spans) { this->spans = spans; }

        DDPatchDataFieldSpanPointer<T> extract() {
            DDPatchDataFieldSpanPointer<T> spans = std::exchange(this->spans, {});
            return spans;
        }

        inline virtual void free_alloc() { spans = {}; }
    };
} // namespace shamrock::solvergraph

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
 * @file Field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class Field : public FieldRefs<T> {

        // TODO In the long run this class should become what was compute field

        u32 nvar;
        std::string name;
        ComputeField<T> field;

        public:
        Field(u32 nvar, std::string name, std::string texsymbol)
            : nvar(nvar), name(name), FieldRefs<T>(name, texsymbol) {}

        void sync_refs() {
            this->set_ref_sync_spans(
                field.field_data.template map<std::reference_wrapper<PatchDataField<T>>>(
                    [&](u64 id, PatchDataField<T> &pdf) {
                        return std::ref(pdf);
                    }));
        }

        // overload only the non const case
        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {

            auto new_patchdatafield = [&](u32 size) {
                auto ret = PatchDataField<T>(name, nvar);
                ret.resize(size);
                return ret;
            };

            auto ensure_patchdatafield_sizes = [&](u32 size, auto &pdatfield) {
                if (pdatfield.get_obj_cnt() != size) {
                    pdatfield.resize(size);
                }
            };

            on_distributeddata_diff(
                this->spans,
                sizes,
                [&](u64 id) {
                    field.field_data.add_obj(id, new_patchdatafield(sizes.get(id)));
                },
                [&](u64 id) {
                    ensure_patchdatafield_sizes(sizes.get(id), field.field_data.get(id));
                },
                [&](u64 id) {
                    field.field_data.erase(id);
                });

            sync_refs();
        }

        inline ComputeField<T> extract() {
            this->set_ref_sync_spans({});
            return std::move(field);
        }

        inline sham::DeviceBuffer<T> &get_buf(u64 id_patch) {
            return field.field_data.get(id_patch).get_buf();
        }

        inline PatchDataField<T> &get_field(u64 id_patch) { return field.field_data.get(id_patch); }
    };
} // namespace shamrock::solvergraph

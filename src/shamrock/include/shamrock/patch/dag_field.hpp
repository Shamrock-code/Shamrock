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
 * @file dag_field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambase/DistributedData.hpp"
#include "shamdag/IDataEdge.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include <functional>
#include <memory>
#include <optional>

namespace shamrock::dag {

    // Like here the idea is that we will attach edges to fields buffers, they will never own it.
    // I should add the possibility to resize PatchDataField from edges and also a way to regen the
    // fields_refs then ...

    template<class T, u32 nvar = dynamic_nvar, bool pointer_access = access_t_span>
    class FieldRef : public IDataEdge {

        public:
        using FieldRef_t = shambase::DistributedData<PatchDataFieldSpan<T, nvar, pointer_access>>;

        FieldRef(
            std::string name,
            std::string texsymbol,
            std::optional<std::function<FieldRef_t()>> _get_field_refs,
            std::optional<std::function<void(shambase::DistributedData<u32>)>> _require_field_sizes)
            : name(name), texsymbol(texsymbol), _get_field_refs(_get_field_refs),
              _require_field_sizes(_require_field_sizes) {}

        inline std::string _impl_get_label() { return name; };
        inline std::string _impl_get_tex_symbol() { return texsymbol; };

        static auto
        attach_to_compute_field(std::string name, std::string texsymbol, ComputeField<T> &field) {
            auto field_ref = std::make_shared<FieldRef<T, nvar, pointer_access>>(name, texsymbol);

            // TODO: implement lambdas

            return field_ref;
        }

        FieldRef_t &get_field_refs() { return _get_field_refs(); }

        private:
        std::string name;
        std::string texsymbol;

        std::optional<std::function<FieldRef_t()>> _get_field_refs;
        std::optional<std::function<void(shambase::DistributedData<u32>)>> _require_field_sizes;
    };

} // namespace shamrock::dag

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
 * @file SGDirectPlummer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <sycl/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec>
    class SGDirectPlummer : public shamrock::solvergraph::INode {

        using Tscal          = shambase::VecComponent<Tvec>;
        using PatchDataField = ::PatchDataField<Tvec>;

        Tscal epsilon;       ///< Gravitational softening length
        bool reference_mode; ///< If true this is a double loopm for debugging purposes

        public:
        explicit SGDirectPlummer(Tscal epsilon, bool reference_mode = false)
            : epsilon(epsilon), reference_mode(reference_mode) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IDataEdge<Tscal> &gpart_mass;
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shamrock::solvergraph::FieldRefs<Tvec> &field_xyz;
            shamrock::solvergraph::FieldRefs<Tvec> &field_axyz_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> gpart_mass,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> field_xyz,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> field_axyz_ext) {
            __internal_set_ro_edges({sizes, gpart_mass, constant_G, field_xyz});
            __internal_set_rw_edges({field_axyz_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tvec>>(3),
                get_rw_edge<shamrock::solvergraph::FieldRefs<Tvec>>(0)};
        }

        inline std::string _impl_get_label()const override { return "SGDirectPlummer"; }
        std::string _impl_get_tex()const override { return "TODO"; }

        protected:
        void _impl_evaluate_internal() override;
    };

} // namespace shammodels::sph::modules

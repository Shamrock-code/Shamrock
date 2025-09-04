// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddForceCentralGravPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {

    template<class Tvec>
    class AddForceCentralGravPotential : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceCentralGravPotential() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shamrock::solvergraph::IDataEdge<Tscal> &central_mass;
            const shamrock::solvergraph::IDataEdge<Tvec> &central_pos;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> central_mass,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tvec>> central_pos,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges(
                {constant_G, central_mass, central_pos, spans_positions, sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(4),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal() {

            [[maybe_unused]] StackEntry stack_loc{};

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal cmass = edges.central_mass.data;
            Tscal G     = edges.constant_G.data;
            Tvec cpos   = edges.central_pos.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_positions.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [mGM = -cmass * G, cpos](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                    Tvec r_a       = xyz[gid] - cpos;
                    Tscal abs_ra   = sycl::length(r_a);
                    Tscal abs_ra_3 = abs_ra * abs_ra * abs_ra;
                    axyz_ext[gid] += mGM * r_a / abs_ra_3;
                });
        }

        inline virtual std::string _impl_get_label() { return "AddForceCentralGravPotential"; };

        inline virtual std::string _impl_get_tex() {

            auto constant_G   = get_ro_edge_base(0).get_tex_symbol();
            auto central_mass = get_ro_edge_base(1).get_tex_symbol();
            auto central_pos  = get_ro_edge_base(2).get_tex_symbol();
            auto positions    = get_ro_edge_base(3).get_tex_symbol();
            auto axyz_ext     = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Add force (central gravitational potential)

                \begin{align}
                {axyz_ext}_i = -{constant_G} * {central_mass} * {central_pos}_i / {positions}_i^3
                \end{align}
            )tex";

            shambase::replace_all(tex, "{constant_G}", constant_G);
            shambase::replace_all(tex, "{central_mass}", central_mass);
            shambase::replace_all(tex, "{central_pos}", central_pos);
            shambase::replace_all(tex, "{positions}", positions);
            shambase::replace_all(tex, "{axyz_ext}", axyz_ext);

            return tex;
        }
    };

} // namespace shammodels::common::modules

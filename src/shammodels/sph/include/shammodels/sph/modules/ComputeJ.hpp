// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeJ.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class NodeComputeJ : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;
        Tscal part_mass;
        Tscal mu_0;
        Tscal c;

        public:
        NodeComputeJ(Tscal part_mass, Tscal mu_0, Tscal c)
            : part_mass(part_mass), mu_0(mu_0), c(c) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
            const shamrock::solvergraph::IFieldSpan<Tscal> &omega;
            const shamrock::solvergraph::IFieldSpan<Tvec> &B_on_rho;
            shamrock::solvergraph::IFieldSpan<Tvec> &J;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> xyz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> hpart,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> omega,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> B_on_rho,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> J) {
            __internal_set_ro_edges({part_counts, neigh_cache, xyz, hpart, omega, B_on_rho});
            __internal_set_rw_edges({J});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(5),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeJ"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

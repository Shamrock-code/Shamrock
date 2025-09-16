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
 * @file ComputeOmega.hpp
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

    template<class Tvec, class SPHKernel>
    class NodeComputeOmega : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal kernel_radius = SPHKernel::Rkern;
        Tscal part_mass;

        public:
        NodeComputeOmega(Tscal part_mass) : part_mass(part_mass) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
            shamrock::solvergraph::IFieldSpan<Tscal> &omega;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> xyz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> hpart,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> omega) {
            __internal_set_ro_edges({part_counts, neigh_cache, xyz, hpart});
            __internal_set_rw_edges({omega});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ComputeOmega"; };

        virtual std::string _impl_get_tex();
    };

    template<class Tvec, template<class> class SPHKernel>
    class ComputeOmega {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        ComputeOmega(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void compute_omega();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules

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
 * @file Substepping.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the substeps of the RESPA algorithm as a solver graph node.
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>

namespace shammodels::sph::modules {
    template<class Tvec, template<class> class SPHKernel>
    class Substepping : public shamrock::solvergraph::INode {

        using Tscal    = shambase::VecComponent<Tvec>;
        using Solver   = sph::Solver<Tvec, SPHKernel>;
        using Config   = SolverConfig<Tvec, SPHKernel>;
        using u_morton = typename Config::u_morton;
        std::shared_ptr<INode> do_foward_euler_vxyz_ptr;
        std::shared_ptr<INode> do_foward_euler_u_ptr;
        std::shared_ptr<INode> do_foward_euler_xyz_ptr;
        Config solver_config;
        ShamrockCtx &ctx;
        SolverStorage<Tvec, u_morton> storage{};

        public:
        Substepping(
            Config &solver_config,
            ShamrockCtx &ctx,
            std::shared_ptr<INode> do_foward_euler_vxyz_ptr,
            std::shared_ptr<INode> do_foward_euler_u_ptr,
            std::shared_ptr<INode> do_foward_euler_xyz_ptr)
            : solver_config(solver_config), ctx(ctx),
              do_foward_euler_vxyz_ptr(do_foward_euler_vxyz_ptr),
              do_foward_euler_u_ptr(do_foward_euler_u_ptr),
              do_foward_euler_xyz_ptr(do_foward_euler_xyz_ptr) {}

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &current_time;
            const shamrock::solvergraph::IDataEdge<Tscal> &dt_sph;
            const shamrock::solvergraph::IDataEdge<Tscal> &dt_force;
            const shamrock::solvergraph::Indexes<u32> &sizes;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> current_time,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> dt_sph,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> dt_force,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes) {
            __internal_set_ro_edges({current_time, dt_sph, dt_force, sizes});
            __internal_set_rw_edges({});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(3)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "Substepping"; };

        virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::sph::modules

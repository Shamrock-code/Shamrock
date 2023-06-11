// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
namespace shammodels {

    template<class Tvec>
    struct GridEulerSolverConfig {};

    template<class Tvec, class TgridVec>
    class GridEulerSolver {

        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u32;

        using Config = GridEulerSolverConfig<Tvec>;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
        }

        // serial patch tree control
        std::unique_ptr<SerialPatchTree<TgridVec>> sptree;
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { sptree.reset(); }

        GridEulerSolver(ShamrockCtx &context) : context(context) {}
    };

} // namespace shammodels
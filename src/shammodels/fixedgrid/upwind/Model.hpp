// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/fixedgrid/upwind/Solver.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::fgrid::upwind {

    template<class Tvec, class TgridVec>
    class Model {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        ShamrockCtx &ctx;

        using Solver = Solver<Tvec, TgridVec>;
        Solver solver;

        Model(ShamrockCtx &ctx) : ctx(ctx), solver(ctx){};
    };

} // namespace shammodels::fgrid::upwind
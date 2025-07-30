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
 * @file CartesianRender.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    class CartesianRender {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        CartesianRender(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        using field_getter_t = const sham::DeviceBuffer<Tfield> &(
            const shamrock::patch::Patch cur_p, shamrock::patch::PatchData &pdat);

        sham::DeviceBuffer<Tfield> compute_slice(
            std::function<field_getter_t> field_getter,
            Tvec center,
            Tvec delta_x,
            Tvec delta_y,
            u32 nx,
            u32 ny);

        sham::DeviceBuffer<Tfield> compute_column_integ(
            std::function<field_getter_t> field_getter,
            Tvec center,
            Tvec delta_x,
            Tvec delta_y,
            u32 nx,
            u32 ny);

        sham::DeviceBuffer<Tfield> compute_slice(
            std::string field_name, Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny);

        sham::DeviceBuffer<Tfield> compute_column_integ(
            std::string field_name, Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules

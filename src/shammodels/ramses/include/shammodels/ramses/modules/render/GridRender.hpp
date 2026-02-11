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
 * @file GridRender.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::ramses::modules {

    template<class Tvec, class TgridVec, class Tfield>
    class GridRender {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = basegodunov::SolverConfig<Tvec, TgridVec>;
        using Storage = basegodunov::SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        GridRender(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        using field_getter_t = const sham::DeviceBuffer<Tfield> &(
            const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat);

        sham::DeviceBuffer<Tfield> compute_slice(
            std::function<field_getter_t> field_getter, const sham::DeviceBuffer<Tvec> &positions);

        sham::DeviceBuffer<Tfield> compute_slice(
            std::string field_name, const sham::DeviceBuffer<Tvec> &positions);

        inline sham::DeviceBuffer<Tfield> compute_slice(
            std::string field_name, const std::vector<Tvec> &positions) {
            sham::DeviceBuffer<Tvec> positions_buf{
                positions.size(), shamsys::instance::get_compute_scheduler_ptr()};
            positions_buf.copy_from_stdvec(positions);
            return compute_slice(field_name, positions_buf);
        }

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::ramses::modules

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
 * @file Model.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/zeus/Solver.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamtree/kernels/geometry_utils.hpp"

namespace shammodels::zeus {

    /**
     * @brief Zeus scheme \cite Zeus2d_main \cite Fargo3D_1 implementation in Shamrock (WIP)
     *
     * The dimension is computed from the input vector types
     *
     * @tparam Tvec vector type used in physical space
     * @tparam TgridVec grid vector type (integer vector)
     */
    template<class Tvec, class TgridVec>
    class Model {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        ShamrockCtx &ctx;

        using Solver = Solver<Tvec, TgridVec>;
        Solver solver;

        Model(ShamrockCtx &ctx) : ctx(ctx), solver(ctx) {};

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// setup function
        ////////////////////////////////////////////////////////////////////////////////////////////

        void init_scheduler(u32 crit_split, u32 crit_merge);

        void make_base_grid(TgridVec bmin, TgridVec cell_size, u32_3 cell_count);

        void dump_vtk(std::string filename);

        Tscal evolve_once(Tscal t_current, Tscal dt_input);

        template<class T>
        inline void set_field_value_lambda(
            std::string field_name, const std::function<T(Tvec, Tvec)> pos_to_val) {

            StackEntry stack_loc{};

            using Block = typename Solver::Config::AMRBlock;

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata([&](u64 patch_id,
                                                    shamrock::patch::PatchDataLayer &pdat) {
                sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
                sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

                PatchDataField<T> &f
                    = pdat.template get_field<T>(sched.pdl_old().get_field_idx<T>(field_name));

                {
                    auto &buf = f.get_buf();
                    auto acc  = buf.copy_to_stdvec();

                    auto cell_min = buf_cell_min.copy_to_stdvec();
                    auto cell_max = buf_cell_max.copy_to_stdvec();

                    Tscal scale_factor = solver.solver_config.grid_coord_to_pos_fact;
                    for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                        Tvec block_min  = cell_min[i].template convert<Tscal>() * scale_factor;
                        Tvec block_max  = cell_max[i].template convert<Tscal>() * scale_factor;
                        Tvec delta_cell = (block_max - block_min) / Block::side_size;

                        Block::for_each_cell_in_block(delta_cell, [&](u32 lid, Tvec delta) {
                            Tvec bmin                        = block_min + delta;
                            acc[i * Block::block_size + lid] = pos_to_val(bmin, bmin + delta_cell);
                        });
                    }

                    buf.copy_from_stdvec(acc);
                }
            });
        }

        inline std::pair<Tvec, Tvec> get_cell_coords(
            std::pair<TgridVec, TgridVec> block_coords, u32 lid) {
            using Block = typename Solver::Config::AMRBlock;
            auto tmp    = Block::utils_get_cell_coords(block_coords, lid);
            tmp.first *= solver.solver_config.grid_coord_to_pos_fact;
            tmp.second *= solver.solver_config.grid_coord_to_pos_fact;
            return tmp;
        }
    };

} // namespace shammodels::zeus

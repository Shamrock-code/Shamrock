// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Model.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::basegodunov {

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

        template<class T>
        inline void set_field_value_lambda(
            std::string field_name, const std::function<T(Tvec, Tvec)> pos_to_val) {

            StackEntry stack_loc{};

            using Block = typename Solver::Config::AMRBlock;

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata([&](u64 patch_id,
                                                    shamrock::patch::PatchData &pdat) {
                sycl::buffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
                sycl::buffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

                PatchDataField<T> &f
                    = pdat.template get_field<T>(sched.pdl.get_field_idx<T>(field_name));
                
                auto f_nvar =  f.get_nvar() / Block::block_size;
                std::cout << " f_nvar = " << f_nvar << "\n\n";

                {
                    auto &buf = shambase::get_check_ref(f.get_buf());
                    sycl::host_accessor acc{buf};

                    sycl::host_accessor cell_min{buf_cell_min, sycl::read_only};
                    sycl::host_accessor cell_max{buf_cell_max, sycl::read_only};

                    Tscal scale_factor = solver.solver_config.grid_coord_to_pos_fact;
                    for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                        Tvec block_min  = cell_min[i].template convert<Tscal>() * scale_factor;
                        Tvec block_max  = cell_max[i].template convert<Tscal>() * scale_factor;
                        Tvec delta_cell = (block_max - block_min) / Block::side_size;

                        Block::for_each_cell_in_block(delta_cell, [&](u32 lid, Tvec delta) {
                            Tvec bmin                        = block_min + delta;
                            acc[(i * Block::block_size + lid) * f_nvar ] = pos_to_val(bmin, bmin + delta_cell);
                        });
                    }
                }
            });
        }

        inline std::pair<Tvec, Tvec>
        get_cell_coords(std::pair<TgridVec, TgridVec> block_coords, u32 lid) {
            using Block = typename Solver::Config::AMRBlock;
            auto tmp    = Block::utils_get_cell_coords(block_coords, lid);
            tmp.first *= solver.solver_config.grid_coord_to_pos_fact;
            tmp.second *= solver.solver_config.grid_coord_to_pos_fact;
            return tmp;
        }

        inline f64 evolve_once_time_expl(f64 t_curr, f64 dt_input) {
            return solver.evolve_once_time_expl(t_curr, dt_input);
        }

        inline void timestep() { solver.evolve_once(); }

        inline void evolve_once() {
            solver.evolve_once();
            solver.print_timestep_logs();
        }

        inline bool evolve_until(Tscal target_time, i32 niter_max) {
            return solver.evolve_until(target_time, niter_max);
        }
    };

} // namespace shammodels::basegodunov

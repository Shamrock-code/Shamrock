// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Solver.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/vec.hpp"
#include "shammodels/amr/AMRBlock.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
namespace shammodels::basegodunov {

    enum RiemmanSolverMode {
        Rusanov = 0,
        HLL = 1
    };

    enum SlopeMode{
        None = 0,
        VanLeer_f = 1,
        VanLeer_std = 2,
        VanLeer_sym = 3,
        Minmod = 4,
    };

    template<class Tvec,class TgridVec>
    struct SolverConfig {

        using Tscal              = shambase::VecComponent<Tvec>;

        Tscal eos_gamma = 5./3.;

        Tscal grid_coord_to_pos_fact = 1;

        static constexpr u32 NsideBlockPow = 1;
        using AMRBlock = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

        inline void set_eos_gamma(Tscal gamma){
            eos_gamma = gamma;
        }

        RiemmanSolverMode riemman_config = HLL;
        SlopeMode slope_config = VanLeer_sym;

        Tscal Csafe = 0.9;
    };

    template<class Tvec, class TgridVec>
    class Solver {public:

        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u64;
        using Config = SolverConfig<Tvec,TgridVec>;

        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        SolverStorage<Tvec,TgridVec, u_morton> storage {};

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
            context.pdata_layout_add_field<Tscal>("rho", AMRBlock::block_size);
            context.pdata_layout_add_field<Tvec>("rhovel", AMRBlock::block_size);
            context.pdata_layout_add_field<Tscal>("rhoetot", AMRBlock::block_size);
        }

        Solver(ShamrockCtx &context) : context(context) {}

        Tscal evolve_once(Tscal t_current,Tscal dt_input);

        void do_debug_vtk_dump(std::string filename);
    };

} // namespace shammodels::basegodunov
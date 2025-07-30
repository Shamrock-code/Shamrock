// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisSodTube.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shammath/riemann.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/ramses/modules/AnalysisSodTube.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::AnalysisSodTube<Tvec, TgridVec>::compute_L2_dist()
    -> field_val {

    auto get_gamma = [&]() -> Tscal {
        return solver_config.eos_gamma;
        return {};
    };

    Tscal gamma = get_gamma();

    auto &sched = scheduler();

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    Tscal sum_L2_rho = 0;
    Tvec sum_L2_v    = {0, 0, 0};
    Tscal sum_L2_P   = 0;
    Tscal N          = 0;

    Tscal one_over_Nside = 1. / AMRBlock::Nside;

    Tscal dxfact = solver_config.grid_coord_to_pos_fact;

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<TgridVec> &buf_block_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = pdat.get_field_buf_ref<TgridVec>(1);
            sham::DeviceBuffer<Tscal> &buf_rho          = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov          = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe         = pdat.get_field_buf_ref<Tscal>(irhoetot);

            {
                auto acc_block_min = buf_block_min.copy_to_stdvec();
                auto acc_block_max = buf_block_max.copy_to_stdvec();
                auto acc_rho       = buf_rho.copy_to_stdvec();
                auto acc_rhov      = buf_rhov.copy_to_stdvec();
                auto acc_rhoe      = buf_rhoe.copy_to_stdvec();

                for (u32 i = 0; i < cell_count; i++) {
                    const u32 block_id    = i / AMRBlock::block_size;
                    const u32 cell_loc_id = i % AMRBlock::block_size;

                    TgridVec lower = acc_block_min[block_id];
                    TgridVec upper = acc_block_max[block_id];

                    Tvec lower_flt = lower.template convert<Tscal>() * dxfact;
                    Tvec upper_flt = upper.template convert<Tscal>() * dxfact;

                    Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

                    Tvec xyz   = lower_flt + (block_cell_size / 2);
                    Tscal rho  = acc_rho[i];
                    Tvec rhov  = acc_rhov[i];
                    Tscal rhoe = acc_rhoe[i];

                    auto conststate = shammath::ConsState<Tvec>{rho, rhoe, rhov};
                    auto prim_state = shammath::cons_to_prim(conststate, gamma);

                    Tscal P = prim_state.press;
                    Tvec v  = prim_state.vel;

                    Tscal x = sham::dot(xyz, direction) - x_ref;

                    if ((x + x_ref) > x_min && (x + x_ref) < x_max) {

                        auto result_sod = solution.get_value(time_val, x);

                        Tscal d_rho = rho - result_sod.rho;
                        Tscal d_P   = P - result_sod.P;
                        Tvec d_vxyz = v - result_sod.vx * direction;

                        sum_L2_rho += d_rho * d_rho;
                        sum_L2_P += d_P * d_P;
                        sum_L2_v += d_vxyz * d_vxyz;
                        N += 1;
                    }
                }
            }
        });

    Tscal tot_N = shamalgs::collective::allreduce_sum(N);

    if (tot_N == 0) {
        shambase::throw_with_loc<std::runtime_error>("no particle in wanted region");
    }

    Tscal mpi_sum_L2_P   = shamalgs::collective::allreduce_sum(sum_L2_P) / tot_N;
    Tscal mpi_sum_L2_rho = shamalgs::collective::allreduce_sum(sum_L2_rho) / tot_N;
    Tvec mpi_sum_L2_v    = shamalgs::collective::allreduce_sum(sum_L2_v) / tot_N;

    return field_val{mpi_sum_L2_rho, mpi_sum_L2_v, mpi_sum_L2_P};
}

template class shammodels::basegodunov::modules::AnalysisSodTube<f64_3, i64_3>;

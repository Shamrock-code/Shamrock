// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisSodTube.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>::compute_L2_dist() -> field_val {

    auto get_gamma = [&]() -> Tscal {
        using SolverConfigEOS     = typename Config::EOSConfig;
        using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
        if (SolverEOS_Adiabatic *eos_config
            = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {
            return eos_config->gamma;
        }
        shambase::throw_with_loc<std::invalid_argument>(
            "The sod analysis is only available for adiabatic EOS");
        return {};
    };

    Tscal gamma = get_gamma();

    auto rho_h = [&](Tscal h) {
        return shamrock::sph::rho_h(solver_config.gpart_mass, h, Kernel::hfactd);
    };

    auto &sched = scheduler();

    using namespace shamrock::patch;
    PatchDataLayerLayout &pdl = sched.pdl();

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");

    Tscal sum_L2_rho = 0;
    Tvec sum_L2_v    = {0, 0, 0};
    Tscal sum_L2_P   = 0;
    Tscal N          = 0;

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            auto &xyz_buf   = pdat.get_field_buf_ref<Tvec>(ixyz);
            auto &hpart_buf = pdat.get_field_buf_ref<Tscal>(ihpart);
            auto &vxyz_buf  = pdat.get_field_buf_ref<Tvec>(ivxyz);
            auto &uint_buf  = pdat.get_field_buf_ref<Tscal>(iuint);

            u32 len = pdat.get_obj_cnt();

            {
                auto acc_xyz   = xyz_buf.copy_to_stdvec();
                auto acc_hpart = hpart_buf.copy_to_stdvec();
                auto acc_vxyz  = vxyz_buf.copy_to_stdvec();
                auto acc_uint  = uint_buf.copy_to_stdvec();

                for (u32 i = 0; i < len; i++) {

                    Tvec xyz  = acc_xyz[i];
                    Tscal h   = acc_hpart[i];
                    Tvec vxyz = acc_vxyz[i];
                    Tscal u   = acc_uint[i];

                    Tscal rho = rho_h(h);
                    Tscal P   = shamphys::EOS_Adiabatic<Tscal>::pressure(gamma, rho, u);

                    Tscal x = sham::dot(xyz, direction) - x_ref;

                    if ((x + x_ref) > x_min && (x + x_ref) < x_max) {

                        auto result_sod = solution.get_value(time_val, x);

                        Tscal d_rho = rho - result_sod.rho;
                        Tscal d_P   = P - result_sod.P;
                        Tvec d_vxyz = vxyz - result_sod.vx * direction;

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

using namespace shammath;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M4>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M6>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M8>;

template class shammodels::sph::modules::AnalysisSodTube<f64_3, C2>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, C4>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, C6>;

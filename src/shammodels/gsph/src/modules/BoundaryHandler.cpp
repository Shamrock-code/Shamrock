// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BoundaryHandler.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Boundary condition application implementation
 */

#include "shambase/stacktrace.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/modules/BoundaryHandler.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void BoundaryHandler<Tvec, SPHKernel>::apply_position_boundary(Tscal time_val) {
        StackEntry stack_loc{};

        shamlog_debug_ln("GSPH", "apply position boundary");

        PatchScheduler &sched = scheduler();
        shamrock::SchedulerUtility integrators(sched);
        shamrock::ReattributeDataUtility reatrib(sched);

        auto &pdl         = sched.pdl();
        const u32 ixyz    = pdl.template get_field_idx<Tvec>("xyz");
        const u32 ivxyz   = pdl.template get_field_idx<Tvec>("vxyz");
        auto [bmin, bmax] = sched.template get_box_volume<Tvec>();

        using SolverConfigBC           = typename Config::BCConfig;
        using SolverBCFree             = typename SolverConfigBC::Free;
        using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
        using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

        if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln("PositionUpdated", "free boundaries skipping geometry update");
            }
        } else if (
            SolverBCPeriodic *c
            = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
            integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});
        } else if (
            SolverBCShearingPeriodic *c
            = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
            integrators.fields_apply_shearing_periodicity(
                ixyz,
                ivxyz,
                std::pair{bmin, bmax},
                c->shear_base,
                c->shear_dir,
                c->shear_speed * time_val,
                c->shear_speed);
        } else {
            shambase::throw_with_loc<std::runtime_error>(
                "GSPH: Unsupported boundary condition type.");
        }

        reatrib.reatribute_patch_objects(storage.serial_patch_tree.get(), "xyz");
    }

    // Explicit instantiations
    template class BoundaryHandler<f64_3, shammath::M4>;
    template class BoundaryHandler<f64_3, shammath::M6>;
    template class BoundaryHandler<f64_3, shammath::M8>;
    template class BoundaryHandler<f64_3, shammath::C2>;
    template class BoundaryHandler<f64_3, shammath::C4>;
    template class BoundaryHandler<f64_3, shammath::C6>;
    template class BoundaryHandler<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules

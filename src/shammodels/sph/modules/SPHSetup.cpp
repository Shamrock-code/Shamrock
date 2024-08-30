// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHSetup.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/SPHSetup.hpp"
#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::apply_setup(
    SetupNodePtr setup, bool part_reordering) {

    if (!bool(setup)) {
        shambase::throw_with_loc<std::invalid_argument>("The setup shared pointer is empty");
    }

    shambase::Timer time_setup;
    time_setup.start();
    StackEntry stack_loc{};

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    shamrock::DataInserterUtility inserter(sched);
    u32 insert_step = sched.crit_patch_split * 8;

    while (!setup->is_done()) {

        shamrock::patch::PatchData pdat = setup->next_n(insert_step);

        inserter.push_patch_data<Tvec>(pdat, "xyz", sched.crit_patch_split * 8, [&]() {
            modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(context, solver_config, storage)
                .update_load_balancing();
        });
    }

    if (part_reordering) {
        modules::ParticleReordering<Tvec, u32, SPHKernel>(context, solver_config, storage)
            .reorder_particles();
    }

    time_setup.end();
    if (shamcomm::world_rank() == 0) {
        logger::info_ln("SPH setup", "the setup took :", time_setup.elasped_sec(), "s");
    }
}

using namespace shammath;
template class shammodels::sph::modules::SPHSetup<f64_3, M4>;
template class shammodels::sph::modules::SPHSetup<f64_3, M6>;
template class shammodels::sph::modules::SPHSetup<f64_3, M8>;

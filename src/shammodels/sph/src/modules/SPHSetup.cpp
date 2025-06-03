// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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
#include "shammodels/sph/modules/setup/CombinerAdd.hpp"
#include "shammodels/sph/modules/setup/GeneratorLatticeHCP.hpp"
#include "shammodels/sph/modules/setup/GeneratorMCDisc.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyDiscWarp.hpp"
#include "shammodels/sph/modules/setup/ModifierOffset.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>
shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::make_generator_lattice_hcp(
    Tscal dr, std::pair<Tvec, Tvec> box) {
    return std::shared_ptr<ISPHSetupNode>(new GeneratorLatticeHCP<Tvec>(context, dr, box));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>
shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::make_generator_disc_mc(
    Tscal part_mass,
    Tscal disc_mass,
    Tscal r_in,
    Tscal r_out,
    std::function<Tscal(Tscal)> sigma_profile,
    std::function<Tscal(Tscal)> H_profile,
    std::function<Tscal(Tscal)> rot_profile,
    std::function<Tscal(Tscal)> cs_profile,
    std::mt19937 eng,
    Tscal init_h_factor) {
    return std::shared_ptr<ISPHSetupNode>(new GeneratorMCDisc<Tvec, SPHKernel>(
        context,
        solver_config,
        part_mass,
        disc_mass,
        r_in,
        r_out,
        sigma_profile,
        H_profile,
        rot_profile,
        cs_profile,
        eng,
        init_h_factor));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>
shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::make_combiner_add(
    SetupNodePtr parent1, SetupNodePtr parent2) {
    return std::shared_ptr<ISPHSetupNode>(new CombinerAdd<Tvec>(context, parent1, parent2));
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::apply_setup(
    SetupNodePtr setup, bool part_reordering, std::optional<u32> insert_step) {

    if (!bool(setup)) {
        shambase::throw_with_loc<std::invalid_argument>("The setup shared pointer is empty");
    }

    shambase::Timer time_setup;
    time_setup.start();
    StackEntry stack_loc{};

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    shamrock::DataInserterUtility inserter(sched);
    u32 _insert_step = sched.crit_patch_split * 8;
    if (bool(insert_step)) {
        _insert_step = insert_step.value();
    }

    while (!setup->is_done()) {

        shamrock::patch::PatchData pdat = setup->next_n(_insert_step);

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

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>
shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::make_modifier_warp_disc(
    SetupNodePtr parent, Tscal Rwarp, Tscal Hwarp, Tscal inclination, Tscal posangle) {
    return std::shared_ptr<ISPHSetupNode>(new ModifierApplyDiscWarp<Tvec, SPHKernel>(
        context, solver_config, parent, Rwarp, Hwarp, inclination, posangle));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>
shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::make_modifier_add_offset(
    SetupNodePtr parent, Tvec offset_postion, Tvec offset_velocity) {

    return std::shared_ptr<ISPHSetupNode>(
        new ModifierOffset<Tvec, SPHKernel>(context, parent, offset_postion, offset_velocity));
}

using namespace shammath;
template class shammodels::sph::modules::SPHSetup<f64_3, M4>;
template class shammodels::sph::modules::SPHSetup<f64_3, M6>;
template class shammodels::sph::modules::SPHSetup<f64_3, M8>;

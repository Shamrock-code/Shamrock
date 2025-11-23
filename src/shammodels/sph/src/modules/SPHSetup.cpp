// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHSetup.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @author David Fang (david.fang@ikmail.com)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shammodels/sph/modules/SPHSetup.hpp"
#include "shammodels/sph/modules/setup/CombinerAdd.hpp"
#include "shammodels/sph/modules/setup/GeneratorLatticeHCP.hpp"
#include "shammodels/sph/modules/setup/GeneratorLatticeHCP_smap_sphere.hpp"
#include "shammodels/sph/modules/setup/GeneratorMCDisc.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyCustomWarp.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyDiscWarp.hpp"
#include "shammodels/sph/modules/setup/ModifierFilter.hpp"
#include "shammodels/sph/modules/setup/ModifierOffset.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"
#include "shamsys/NodeInstance.hpp"

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_generator_lattice_hcp(Tscal dr, std::pair<Tvec, Tvec> box) {
    return std::shared_ptr<ISPHSetupNode>(new GeneratorLatticeHCP<Tvec>(context, dr, box));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_generator_lattice_hcp_smap_sphere(
        Tscal dr, Tvec center, Tscal xmax, std::function<Tscal(Tscal)> rhoprofile) {
    std::pair<Tvec, Tvec> box
        = {center + Tvec(-xmax, -xmax, -xmax), center + Tvec(xmax, xmax, xmax)};
    return std::shared_ptr<ISPHSetupNode>(new GeneratorLatticeHCP_smap_sphere<Tvec, SPHKernel>(
        context, solver_config, dr, box, rhoprofile));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_generator_disc_mc(
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
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_combiner_add(SetupNodePtr parent1, SetupNodePtr parent2) {
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

    auto compute_load = [&]() {
        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(context, solver_config, storage)
            .update_load_balancing();
    };

    shamrock::DataInserterUtility inserter(sched);
    u32 _insert_step = sched.crit_patch_split * 8;
    if (bool(insert_step)) {
        _insert_step = insert_step.value();
    }

    while (!setup->is_done()) {

        shamrock::patch::PatchDataLayer pdat = setup->next_n(_insert_step);

        if (solver_config.track_particles_id) {
            // This bit set the tracking id of the particles
            // But be carefull this assume that the particle injection order
            // is independant from the MPI world size. It should be the case for most setups
            // but some generator could miss this assumption.
            // If that is the case please report the issue

            u64 loc_inj = pdat.get_obj_cnt();

            u64 offset_init = 0;
            shamcomm::mpi::Exscan(
                &loc_inj, &offset_init, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

            // we must add the number of already injected part such that the
            // offset start at the right spot.
            // The only thing that bothers me is that this can not handle the case where multiple
            // setups of things like that are applied. But in principle no sane person would do such
            // a thing...
            offset_init += injected_parts;

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            auto &q        = shambase::get_check_ref(dev_sched).get_queue();

            if (loc_inj > 0) {
                sham::DeviceBuffer<u64> part_ids(loc_inj, dev_sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{},
                    sham::MultiRef{part_ids},
                    loc_inj,
                    [offset_init](u32 i, u64 *__restrict part_ids) {
                        part_ids[i] = i + offset_init;
                    });

                pdat.get_field<u64>(pdat.pdl().get_field_idx<u64>("part_id"))
                    .overwrite(part_ids, loc_inj);
            }
        }

        u64 injected
            = inserter.push_patch_data<Tvec>(pdat, "xyz", sched.crit_patch_split * 8, compute_load);

        injected_parts += injected;
    }

    u32 final_balancing_steps = 3;
    for (u32 i = 0; i < final_balancing_steps; i++) {
        ON_RANK_0(
            logger::info_ln(
                "SPH setup", "Final load balancing step", i, "of", final_balancing_steps));
        inserter.balance_load(compute_load);
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
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_modifier_warp_disc(
        SetupNodePtr parent, Tscal Rwarp, Tscal Hwarp, Tscal inclination, Tscal posangle) {
    return std::shared_ptr<ISPHSetupNode>(new ModifierApplyDiscWarp<Tvec, SPHKernel>(
        context, solver_config, parent, Rwarp, Hwarp, inclination, posangle));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_modifier_custom_warp(
        SetupNodePtr parent,
        std::function<Tscal(Tscal)> inc_profile,
        std::function<Tscal(Tscal)> psi_profile,
        std::function<Tvec(Tscal)> k_profile) {
    return std::shared_ptr<ISPHSetupNode>(new ModifierApplyCustomWarp<Tvec, SPHKernel>(
        context, solver_config, parent, inc_profile, psi_profile, k_profile));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_modifier_add_offset(
        SetupNodePtr parent, Tvec offset_postion, Tvec offset_velocity) {

    return std::shared_ptr<ISPHSetupNode>(
        new ModifierOffset<Tvec>(context, parent, offset_postion, offset_velocity));
}

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::SPHSetup<
    Tvec,
    SPHKernel>::make_modifier_filter(SetupNodePtr parent, std::function<bool(Tvec)> filter) {

    return std::shared_ptr<ISPHSetupNode>(
        new ModifierFilter<Tvec, SPHKernel>(context, parent, filter));
}

using namespace shammath;
template class shammodels::sph::modules::SPHSetup<f64_3, M4>;
template class shammodels::sph::modules::SPHSetup<f64_3, M6>;
template class shammodels::sph::modules::SPHSetup<f64_3, M8>;

template class shammodels::sph::modules::SPHSetup<f64_3, C2>;
template class shammodels::sph::modules::SPHSetup<f64_3, C4>;
template class shammodels::sph::modules::SPHSetup<f64_3, C6>;

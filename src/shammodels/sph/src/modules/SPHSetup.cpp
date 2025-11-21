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
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
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
#include "shammodels/sph/modules/setup/GeneratorMCDisc.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyCustomWarp.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyDiscWarp.hpp"
#include "shammodels/sph/modules/setup/ModifierFilter.hpp"
#include "shammodels/sph/modules/setup/ModifierOffset.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include <mpi.h>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> shammodels::sph::modules::
    SPHSetup<Tvec, SPHKernel>::make_generator_lattice_hcp(Tscal dr, std::pair<Tvec, Tvec> box) {
    return std::shared_ptr<ISPHSetupNode>(new GeneratorLatticeHCP<Tvec>(context, dr, box));
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

    auto has_pdat = [&]() {
        bool ret = false;
        using namespace shamrock::patch;
        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            ret = true;
        });
        return ret;
    };

    shamrock::DataInserterUtility inserter(sched);
    u32 _insert_step = sched.crit_patch_split * 8;
    if (bool(insert_step)) {
        _insert_step = insert_step.value();
    }

    while (!setup->is_done()) {

        shamrock::patch::PatchDataLayer pdat = setup->next_n((has_pdat()) ? _insert_step : 0);

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
void shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::apply_setup_new(
    SetupNodePtr setup, bool part_reordering, std::optional<u32> insert_step) {

    if (!bool(setup)) {
        shambase::throw_with_loc<std::invalid_argument>("The setup shared pointer is empty");
    }

    PatchScheduler &sched = shambase::get_check_ref(context.sched);
    shamrock::DataInserterUtility inserter(sched);

    u32 _insert_step = sched.crit_patch_split * 8;
    if (bool(insert_step)) {
        _insert_step = insert_step.value();
    }

    auto compute_load = [&]() {
        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(context, solver_config, storage)
            .update_load_balancing();
    };

    auto has_pdat = [&]() {
        bool ret = false;
        using namespace shamrock::patch;
        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            ret = true;
        });
        return ret;
    };

    shambase::Timer time_part_gen;
    time_part_gen.start();

    shamrock::patch::PatchDataLayer to_insert(sched.get_layout_ptr());

    while (!setup->is_done()) {

        shamrock::patch::PatchDataLayer tmp = setup->next_n(_insert_step);

        if (solver_config.track_particles_id) {
            // This bit set the tracking id of the particles
            // But be carefull this assume that the particle injection order
            // is independant from the MPI world size. It should be the case for most setups
            // but some generator could miss this assumption.
            // If that is the case please report the issue

            u64 loc_inj = tmp.get_obj_cnt();

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

                tmp.get_field<u64>(tmp.pdl().get_field_idx<u64>("part_id"))
                    .overwrite(part_ids, loc_inj);
            }
        }

        to_insert.insert_elements(tmp);

        u64 sum_push = shamalgs::collective::allreduce_sum<u64>(tmp.get_obj_cnt());
        u64 sum_all  = shamalgs::collective::allreduce_sum<u64>(to_insert.get_obj_cnt());

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "SPH setup",
                shambase::format(
                    "generating particles, Nstep = {} ( {:e} ) Ntotal = {} ( {:e} )",
                    sum_push,
                    f64(sum_push),
                    sum_all,
                    f64(sum_all)));
        }

        injected_parts += sum_push;
    }

    time_part_gen.end();
    if (shamcomm::world_rank() == 0) {
        logger::info_ln(
            "SPH setup", "the generation step took :", time_part_gen.elasped_sec(), "s");
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln(
            "SPH setup",
            "final particle count =",
            to_insert.get_obj_cnt(),
            "begining injection ...");
    }

    // injection part (holy shit this is hard)

    shambase::Timer time_part_inject;
    time_part_inject.start();

    while (!shamalgs::collective::are_all_rank_true(to_insert.is_empty(), MPI_COMM_WORLD)) {

        // assume that the sched is synchronized and that there is at least a patch.
        // TODO actually check that

        enum class strategy { RingRotation } mode = strategy::RingRotation;

        if (mode == strategy::RingRotation) {
            // ring rotation strategy

            using namespace shamrock::patch;

            // inject in local domains first
            PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();
            sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

                PatchDataField<Tvec> &xyz = to_insert.get_field<Tvec>(0);

                auto ids = xyz.get_ids_where(
                    [](auto access, u32 id, shammath::CoordRange<Tvec> patch_coord) {
                        Tvec tmp = access[id];
                        return patch_coord.contain_pos(tmp);
                    },
                    patch_coord);

                if (ids.get_size() > _insert_step) {
                    ids.resize(_insert_step);
                }

                if (ids.get_size() > 0) {
                    to_insert.extract_elements(ids, pdat);
                }
            });

            sched.check_patchdata_locality_corectness();

            // rotate the ring
            shambase::Timer time_rotation;
            time_rotation.start();
            {
                using namespace shambase;
                DistributedDataShared<PatchDataLayer> part_exchange = {};

                auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
                auto &q        = shambase::get_check_ref(dev_sched).get_queue();

                PatchDataLayer to_rotate(sched.get_layout_ptr());

                u32 part_to_rotate_size = sham::min(_insert_step / 8, to_insert.get_obj_cnt());

                if (part_to_rotate_size > 0) {
                    sham::DeviceBuffer<u32> part_to_rotate
                        = sham::DeviceBuffer<u32>(part_to_rotate_size, dev_sched);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{},
                        sham::MultiRef{part_to_rotate},
                        part_to_rotate.get_size(),
                        [](u32 i, u32 *__restrict part_to_rotate) {
                            part_to_rotate[i] = i;
                        });

                    to_insert.extract_elements(part_to_rotate, to_rotate);
                }

                auto serialize = [](PatchDataLayer &pdat) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
                    ser.allocate(pdat.serialize_buf_byte_size());
                    pdat.serialize_buf(ser);
                    return ser.finalize();
                };
                auto deserialize = [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the
                    // serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));
                    return PatchDataLayer::deserialize_buf(ser, sched.get_layout_ptr());
                };

                auto ser_sed   = serialize(to_rotate);
                auto send_data = shamcomm::CommunicationBuffer(ser_sed, dev_sched);

                MPI_Request send_rq;
                MPI_Request recv_rq;

                shamcomm::mpi::Isend(
                    send_data.get_ptr(),
                    send_data.get_size(),
                    MPI_BYTE,
                    (shamcomm::world_rank() + 1) % shamcomm::world_size(),
                    0,
                    MPI_COMM_WORLD,
                    &send_rq);

                MPI_Status st;
                i32 cnt;
                shamcomm::mpi::Probe(
                    (shamcomm::world_rank() + shamcomm::world_size() - 1) % shamcomm::world_size(),
                    0,
                    MPI_COMM_WORLD,
                    &st);
                shamcomm::mpi::Get_count(&st, MPI_BYTE, &cnt);

                auto recv_data = shamcomm::CommunicationBuffer(cnt, dev_sched);

                shamcomm::mpi::Irecv(
                    recv_data.get_ptr(),
                    cnt,
                    MPI_BYTE,
                    (shamcomm::world_rank() + shamcomm::world_size() - 1) % shamcomm::world_size(),
                    0,
                    MPI_COMM_WORLD,
                    &recv_rq);

                std::vector<MPI_Status> st_lst(2);
                shamcomm::mpi::Wait(&send_rq, &st_lst[0]);
                shamcomm::mpi::Wait(&recv_rq, &st_lst[1]);

                sham::DeviceBuffer<u8> buf
                    = shamcomm::CommunicationBuffer::convert_usm(std::move(recv_data));

                PatchDataLayer tmp = deserialize(std::move(buf));

                to_insert.insert_elements(tmp);
            }

            time_rotation.end();
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "SPH setup", "the rotation step took :", time_rotation.elasped_sec(), "s");
            }

            inserter.balance_load(compute_load);

        } else {
            shambase::throw_unimplemented("Not implemented yet");
        }

        u64 sum_all = shamalgs::collective::allreduce_sum<u64>(to_insert.get_obj_cnt());

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "SPH setup",
                shambase::format(
                    "injection step, injected {} / {} => {}%",
                    injected_parts - sum_all,
                    injected_parts,
                    f64(injected_parts - sum_all) / f64(injected_parts) * 100.0));
        }
    }

    time_part_inject.end();
    if (shamcomm::world_rank() == 0) {
        logger::info_ln(
            "SPH setup", "the injection step took :", time_part_inject.elasped_sec(), "s");
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

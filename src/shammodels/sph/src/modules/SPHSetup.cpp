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
#include "shambase/tabulate.hpp"
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

struct SetupLog {
    struct State {
        std::vector<u64> count_per_rank;
        std::vector<std::tuple<u32, u32, u64>> msg_list;
    } state;

    u64 step_counter = 0;

    nlohmann::json json_data = nlohmann::json::array();

    void log_state() {
        nlohmann::json step_data;
        step_data["step_counter"]   = step_counter;
        step_data["count_per_rank"] = state.count_per_rank;
        step_data["msg_list"]       = state.msg_list;
        json_data.push_back(step_data);
    }

    void dump_state() {
        std::string fname = "setup_log_step.json";
        if (shamcomm::world_rank() == 0) {
            logger::normal_ln("SPH setup", "dumping setup log to ", fname);
        }

        std::ofstream file(fname);
        file << json_data.dump(4);
        file.close();

        step_counter++;
    }

    void update_count_per_rank(u64 count) {
        std::vector<u64> tmp{count};
        std::vector<u64> recv_count_per_rank;
        shamalgs::collective::vector_allgatherv(tmp, recv_count_per_rank, MPI_COMM_WORLD);
        state.count_per_rank = recv_count_per_rank;
        log_state();
        if (step_counter % 20 == 0)
            dump_state();
    }

    void update_msg_list(std::vector<std::tuple<u32, u32, u64>> &msg_list) {
        state.msg_list = msg_list;
        log_state();
        if (step_counter % 20 == 0)
            dump_state();
    }
};

inline constexpr f64 golden_number = 1.61803398874989484820458683436563;

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>::apply_setup_new(
    SetupNodePtr setup,
    bool part_reordering,
    std::optional<u32> gen_count_per_step,
    std::optional<u32> insert_count_per_step,
    std::optional<u64> max_msg_count_per_rank_per_step,
    std::optional<u64> max_data_count_per_rank_per_step,
    bool do_setup_log) {

    __shamrock_stack_entry();

    if (!bool(setup)) {
        shambase::throw_with_loc<std::invalid_argument>("The setup shared pointer is empty");
    }

    std::optional<SetupLog> setup_log
        = (do_setup_log) ? std::make_optional<SetupLog>() : std::nullopt;

    shambase::Timer time_setup;
    time_setup.start();
    PatchScheduler &sched = shambase::get_check_ref(context.sched);
    shamrock::DataInserterUtility inserter(sched);

    u32 insert_step = sched.crit_patch_split * 8;
    if (bool(insert_count_per_step)) {
        insert_step = insert_count_per_step.value();
    }

    u32 gen_step = sched.crit_patch_split;
    if (bool(gen_count_per_step)) {
        gen_step = gen_count_per_step.value();
    }

    u64 msg_limit = 16;
    if (bool(max_msg_count_per_rank_per_step)) {
        msg_limit = max_msg_count_per_rank_per_step.value();
    }
    u64 data_count_limit = 2 * insert_step;
    if (bool(max_data_count_per_rank_per_step)) {
        data_count_limit = max_data_count_per_rank_per_step.value();
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

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("SPH setup", "generating particles ...");
    }

    while (!setup->is_done()) {
        shambase::Timer timer_gen;
        timer_gen.start();

        shamrock::patch::PatchDataLayer tmp = setup->next_n(gen_step);

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

        timer_gen.end();

        if (shamcomm::world_rank() == 0) {
            f64 part_per_sec = f64(sum_push) / f64(timer_gen.elasped_sec());
            logger::normal_ln(
                "SPH setup",
                shambase::format(
                    "Nstep = {} ( {:.1e} ) Ntotal = {} ( {:.1e} ) rate = {:e} "
                    "N.s^-1",
                    sum_push,
                    f64(sum_push),
                    sum_all,
                    f64(sum_all),
                    part_per_sec));
        }

        if (setup_log) {
            setup_log.value().update_count_per_rank(to_insert.get_obj_cnt());
        }

        injected_parts += sum_push;
    }

    time_part_gen.end();
    if (shamcomm::world_rank() == 0) {
        logger::normal_ln(
            "SPH setup", "the generation step took :", time_part_gen.elasped_sec(), "s");
    }

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln(
            "SPH setup", "final particle count =", injected_parts, "begining injection ...");
    }

    sham::MemPerfInfos mem_perf_infos_start = sham::details::get_mem_perf_info();
    f64 mpi_timer_start                     = shamcomm::mpi::get_timer("total");

    // injection part (holy shit this is hard)

    shambase::Timer time_part_inject;
    time_part_inject.start();

    auto log_inject_status = [&](std::string log_suffix = "") {
        u64 sum_all = shamalgs::collective::allreduce_sum<u64>(to_insert.get_obj_cnt());

        u32 rank_without_patch
            = shamalgs::collective::allreduce_sum<u32>(sched.patch_list.local.size() == 0 ? 1 : 0);

        if (shamcomm::world_rank() == 0) {
            logger::normal_ln(
                "SPH setup",
                shambase::format(
                    "injected {:12} / {:} => {:5.1f}% | ranks with patchs = {:d} / {:d} {}",
                    injected_parts - sum_all,
                    injected_parts,
                    f64(injected_parts - sum_all) / f64(injected_parts) * 100.0,
                    shamcomm::world_size() - rank_without_patch,
                    shamcomm::world_size(),
                    log_suffix));
        }

        if (setup_log) {
            setup_log.value().update_count_per_rank(to_insert.get_obj_cnt());
        }
    };

    auto inject_in_local_domains =
        [&sched, &inserter, &compute_load, &insert_step, &log_inject_status](
            shamrock::patch::PatchDataLayer &to_insert) {
            __shamrock_stack_entry();

            bool has_been_limited = true;

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceBuffer<u32> mask_get_ids_where(0, dev_sched);

            while (has_been_limited) {
                has_been_limited = false;
                using namespace shamrock::patch;

                // inject in local domains first
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();
                sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
                    shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

                    PatchDataField<Tvec> &xyz = to_insert.get_field<Tvec>(0);

                    auto ids = xyz.get_ids_where_recycle_buffer(
                        mask_get_ids_where,
                        [](auto access, u32 id, shammath::CoordRange<Tvec> patch_coord) {
                            Tvec tmp = access[id];
                            return patch_coord.contain_pos(tmp);
                        },
                        patch_coord);

                    if (ids.get_size() > insert_step) {
                        ids.resize(insert_step);
                        has_been_limited = true;
                    }

                    if (ids.get_size() > 0) {
                        to_insert.extract_elements(ids, pdat);
                    }
                });

                sched.check_patchdata_locality_corectness();

                inserter.balance_load(compute_load);

                has_been_limited
                    = !shamalgs::collective::are_all_rank_true(!has_been_limited, MPI_COMM_WORLD);

                if (has_been_limited) {
                    // since we will restart this one let's print
                    log_inject_status(" -> local loop <-");
                }
            }
        };

    auto get_index_per_ranks = [&](f64 &timer_result) {
        __shamrock_stack_entry();

        shambase::Timer time_get_index_per_ranks;
        time_get_index_per_ranks.start();

        SerialPatchTree<Tvec> sptree = SerialPatchTree<Tvec>::build(sched);
        sptree.attach_buf();

        // find where each particle should be inserted
        PatchDataField<Tvec> &pos_field = to_insert.get_field<Tvec>(0);

        if (pos_field.get_nvar() != 1) {
            shambase::throw_unimplemented();
        }

        sycl::buffer<u64> new_id_buf = sptree.compute_patch_owner(
            shamsys::instance::get_compute_scheduler_ptr(),
            pos_field.get_buf(),
            pos_field.get_obj_cnt());

        std::unordered_map<i32, std::vector<u32>> index_per_ranks;
        bool err_id_in_newid = false;
        {
            sycl::host_accessor nid{new_id_buf, sycl::read_only};
            for (u32 i = 0; i < pos_field.get_obj_cnt(); i++) {
                u64 patch_id    = nid[i];
                bool err        = patch_id == u64_max;
                err_id_in_newid = err_id_in_newid || (err);

                i32 rank = sched.get_patch_rank_owner(patch_id);
                index_per_ranks[rank].push_back(i);
            }
        }

        if (err_id_in_newid) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "a new id could not be computed");
        }

        time_get_index_per_ranks.end();
        timer_result = time_get_index_per_ranks.elasped_sec();

        return index_per_ranks;
    };

    f64 total_time_rank_getter = 0;
    f64 max_time_rank_getter   = 0;

    u32 step_count = 0;
    while (!shamalgs::collective::are_all_rank_true(to_insert.is_empty(), MPI_COMM_WORLD)) {

        // assume that the sched is synchronized and that there is at least a patch.
        // TODO actually check that

        using namespace shamrock::patch;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        inject_in_local_domains(to_insert);

        f64 timer_get_index_per_ranks = 0;
        std::unordered_map<i32, std::vector<u32>> index_per_ranks
            = get_index_per_ranks(timer_get_index_per_ranks);
        total_time_rank_getter += timer_get_index_per_ranks;
        max_time_rank_getter = std::max(max_time_rank_getter, timer_get_index_per_ranks);

        // allgather the list of messages
        // format:(u32_2(sender_rank, receiver_rank), u64(indices_size))
        std::vector<u64> send_msg;
        for (auto &[rank, indices] : index_per_ranks) {
            send_msg.push_back(sham::pack32(shamcomm::world_rank(), rank));
            send_msg.push_back(indices.size());
        }

        u64 max_send      = (1 << 24) / shamcomm::world_size();
        bool sync_limited = false;
        if (send_msg.size() > max_send) {
            std::mt19937 eng_local_msg(
                u64(golden_number * 1000 * step_count + shamcomm::world_rank()));
            std::shuffle(send_msg.begin(), send_msg.end(), eng_local_msg);
            send_msg.resize(max_send);
            sync_limited = true;
        }

        std::vector<u64> recv_msg;
        shamalgs::collective::vector_allgatherv(send_msg, recv_msg, MPI_COMM_WORLD);

        std::vector<std::tuple<u32, u32, u64>> msg_list;
        for (u64 i = 0; i < recv_msg.size(); i += 2) {
            u32_2 sender_receiver = sham::unpack32(recv_msg[i]);
            u64 indices_size      = recv_msg[i + 1];

            u32 sender_rank   = sender_receiver.x();
            u32 receiver_rank = sender_receiver.y();

            if (sender_rank == receiver_rank) {
                continue; // only mean that it was not fully inserted in the patch
            }

            msg_list.push_back(std::make_tuple(sender_rank, receiver_rank, indices_size));
        }

        if (setup_log) {
            setup_log.value().update_msg_list(msg_list);
        }

        // shuffle msg_list according to seed golden_number*1000*step_count
        std::mt19937 eng_global_msg(u64(golden_number * 1000 * step_count));
        std::shuffle(msg_list.begin(), msg_list.end(), eng_global_msg);

        // now that we are in sync we can determine who should send to who

        std::vector<u64> msg_count_rank(shamcomm::world_size());
        std::vector<u64> comm_size_rank(shamcomm::world_size());

        std::vector<std::tuple<u32, u32, u64>> rank_msg_list;

        bool was_count_limited = false;
        bool was_size_limited  = false;

        for (auto &[sender_rank, receiver_rank, indices_size] : msg_list) {

            bool msg_count_limit_not_reached = msg_count_rank[receiver_rank] < msg_limit
                                               && msg_count_rank[sender_rank] < msg_limit;

            bool recv_size_limit_not_reached = comm_size_rank[receiver_rank] < data_count_limit
                                               && comm_size_rank[sender_rank] < data_count_limit;

            was_count_limited = was_count_limited || !msg_count_limit_not_reached;
            was_size_limited  = was_size_limited || !recv_size_limit_not_reached;

            bool can_send_recv = msg_count_limit_not_reached && recv_size_limit_not_reached;

            if (can_send_recv) {
                if (sender_rank == shamcomm::world_rank()
                    || receiver_rank == shamcomm::world_rank()) {
                    if (indices_size > 0) {
                        rank_msg_list.push_back(
                            std::make_tuple(sender_rank, receiver_rank, indices_size));
                    }
                }
            }

            msg_count_rank[receiver_rank] += 1;
            msg_count_rank[sender_rank] += 1;
            comm_size_rank[receiver_rank] += indices_size;
            comm_size_rank[sender_rank] += indices_size;
        }

        // logger::raw_ln(
        //     shamcomm::world_rank(),
        //     was_count_limited,
        //     was_size_limited,
        //     msg_count_rank,
        //     comm_size_rank);

        // logger::info_ln(
        //     "SPH setup", "rank", shamcomm::world_rank(), "rank_msg_list", rank_msg_list);

        // extract the data
        shambase::DistributedDataShared<PatchDataLayer> send_data;
        sham::DeviceBuffer idx_to_rem = sham::DeviceBuffer<u32>(0, dev_sched);
        for (auto &[sender_rank, receiver_rank, indices_size] : rank_msg_list) {
            if (sender_rank == shamcomm::world_rank()) {
                std::vector<u32> &idx_to_extract = index_per_ranks[receiver_rank];
                sham::DeviceBuffer _tmp = sham::DeviceBuffer<u32>(idx_to_extract.size(), dev_sched);
                _tmp.copy_from_stdvec(idx_to_extract);

                if (_tmp.get_size() > data_count_limit) {
                    _tmp.resize(data_count_limit);
                }

                PatchDataLayer _tmp_pdat = PatchDataLayer(sched.get_layout_ptr());
                to_insert.append_subset_to(_tmp, _tmp.get_size(), _tmp_pdat);

                idx_to_rem.append(_tmp);

                send_data.add_obj(sender_rank, receiver_rank, std::move(_tmp_pdat));
            }
        }

        to_insert.remove_ids(idx_to_rem, idx_to_rem.get_size());

        // comm the data to the right ranks
        shambase::DistributedDataShared<PatchDataLayer> recv_dat;

        shamalgs::collective::serialize_sparse_comm<PatchDataLayer>(
            dev_sched,
            std::move(send_data),
            recv_dat,
            [&](u64 id) {
                return id; // here the ids in the DDshared are the MPI ranks
            },
            [&](PatchDataLayer &pdat) {
                shamalgs::SerializeHelper ser(dev_sched);
                ser.allocate(pdat.serialize_buf_byte_size());
                pdat.serialize_buf(ser);
                return ser.finalize();
            },
            [&](sham::DeviceBuffer<u8> &&buf) {
                // exchange the buffer held by the distrib data and give it to the
                // serializer
                shamalgs::SerializeHelper ser(dev_sched, std::forward<sham::DeviceBuffer<u8>>(buf));
                return PatchDataLayer::deserialize_buf(ser, sched.get_layout_ptr());
            });

        // insert the data into the data to be inserted
        recv_dat.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
            to_insert.insert_elements(pdat);
        });

        was_count_limited
            = !shamalgs::collective::are_all_rank_true(!was_count_limited, MPI_COMM_WORLD);
        was_size_limited
            = !shamalgs::collective::are_all_rank_true(!was_size_limited, MPI_COMM_WORLD);
        bool was_sync_limited
            = !shamalgs::collective::are_all_rank_true(!sync_limited, MPI_COMM_WORLD);

        std::string log_suffix = "";
        if (was_count_limited) {
            log_suffix += " (msg count limited)";
        }
        if (was_size_limited) {
            log_suffix += " (msg size limited)";
        }
        if (was_sync_limited) {
            log_suffix += " (sync limited)";
        }
        log_inject_status(" <- global loop ->" + log_suffix);

        f64 worst_time_get_index_per_ranks
            = shamalgs::collective::allreduce_max<f64>(timer_get_index_per_ranks);

        step_count++;
    }

    if (setup_log) {
        setup_log.value().dump_state();
    }

    time_part_inject.end();
    if (shamcomm::world_rank() == 0) {
        logger::normal_ln(
            "SPH setup", "the injection step took :", time_part_inject.elasped_sec(), "s");
    }

    sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

    f64 delta_mpi_timer = shamcomm::mpi::get_timer("total") - mpi_timer_start;
    f64 t_dev_alloc
        = (mem_perf_infos_end.time_alloc_device - mem_perf_infos_start.time_alloc_device)
          + (mem_perf_infos_end.time_free_device - mem_perf_infos_start.time_free_device);
    f64 t_host_alloc = (mem_perf_infos_end.time_alloc_host - mem_perf_infos_start.time_alloc_host)
                       + (mem_perf_infos_end.time_free_host - mem_perf_infos_start.time_free_host);

    { // perf infos
        std::vector<f64> time_rank_getter_all_ranks
            = shamalgs::collective::gather(total_time_rank_getter);
        std::vector<f64> max_time_rank_getter_all_ranks
            = shamalgs::collective::gather(max_time_rank_getter);
        std::vector<f64> mpi_timer_all_ranks = shamalgs::collective::gather(delta_mpi_timer);
        std::vector<f64> alloc_time_device_all_ranks = shamalgs::collective::gather(t_dev_alloc);
        std::vector<f64> alloc_time_host_all_ranks   = shamalgs::collective::gather(t_host_alloc);
        std::vector<size_t> max_mem_device_all_ranks
            = shamalgs::collective::gather(mem_perf_infos_end.max_allocated_byte_device);
        std::vector<size_t> max_mem_host_all_ranks
            = shamalgs::collective::gather(mem_perf_infos_end.max_allocated_byte_host);

        if (shamcomm::world_rank() == 0) {
            f64 time_part_inject_sec = time_part_inject.elasped_sec();
            f64 sum_t                = time_part_inject_sec * shamcomm::world_size();

            f64 sum_time_rank_getter = std::accumulate(
                time_rank_getter_all_ranks.begin(), time_rank_getter_all_ranks.end(), 0.0);
            f64 sum_max_time_rank_getter = *std::max_element(
                max_time_rank_getter_all_ranks.begin(), max_time_rank_getter_all_ranks.end());
            f64 sum_mpi
                = std::accumulate(mpi_timer_all_ranks.begin(), mpi_timer_all_ranks.end(), 0.0);
            f64 sum_alloc_device = std::accumulate(
                alloc_time_device_all_ranks.begin(), alloc_time_device_all_ranks.end(), 0.0);
            f64 sum_alloc_host = std::accumulate(
                alloc_time_host_all_ranks.begin(), alloc_time_host_all_ranks.end(), 0.0);
            size_t sum_mem_device_total = std::accumulate(
                max_mem_device_all_ranks.begin(), max_mem_device_all_ranks.end(), 0_u64);
            size_t sum_mem_host_total = std::accumulate(
                max_mem_host_all_ranks.begin(), max_mem_host_all_ranks.end(), 0_u64);

            static constexpr u32 cols_count = 6;

            using Table = shambase::table<cols_count>;

            Table table;

            table.add_double_rule();
            table.add_data(
                {"rank", "rank get (sum/max)", "MPI", "alloc d% h%", "mem (max) d", "mem (max) h"},
                Table::center);
            table.add_double_rule();
            for (u32 i = 0; i < shamcomm::world_size(); i++) {
                table.add_data(
                    {shambase::format("{:<4}", i),
                     shambase::format(
                         "{:.2f}s / {:.2f}s",
                         time_rank_getter_all_ranks[i],
                         max_time_rank_getter_all_ranks[i]),
                     shambase::format("{:.2f}s", mpi_timer_all_ranks[i]),
                     shambase::format(
                         "{:>.1f}% {:<.1f}%",
                         100 * (alloc_time_device_all_ranks[i] / time_part_inject_sec),
                         100 * (alloc_time_host_all_ranks[i] / time_part_inject_sec)),
                     shambase::format("{}", shambase::readable_sizeof(max_mem_device_all_ranks[i])),
                     shambase::format("{}", shambase::readable_sizeof(max_mem_host_all_ranks[i]))},
                    Table::right);
            }
            if (shamcomm::world_size() > 1) {
                table.add_rulled_data({"", "<avg> / <max>", "<avg>", "<avg>", "<sum>", "<sum>"});
                table.add_data(
                    {"all",
                     shambase::format(
                         "{:.2f}s / {:.2f}s",
                         sum_time_rank_getter / shamcomm::world_size(),
                         sum_max_time_rank_getter),
                     shambase::format("{:.2f}s", sum_mpi / shamcomm::world_size()),
                     shambase::format(
                         "{:>.1f}% {:<.1f}%",
                         100 * (sum_alloc_device / sum_t),
                         100 * (sum_alloc_host / sum_t)),
                     shambase::format("{}", shambase::readable_sizeof(sum_mem_device_total)),
                     shambase::format("{}", shambase::readable_sizeof(sum_mem_host_total))},
                    Table::right);
            }
            table.add_rule();
            logger::info_ln("SPH setup", "injection perf report:" + table.render());
        }
    }

    if (part_reordering) {
        modules::ParticleReordering<Tvec, u32, SPHKernel>(context, solver_config, storage)
            .reorder_particles();
    }

    time_setup.end();
    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("SPH setup", "the setup took :", time_setup.elasped_sec(), "s");
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

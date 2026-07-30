// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHSetup.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambase/tabulate.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include "shammodels/gsph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/gsph/modules/GSPHSetup.hpp"
#include "shammodels/gsph/modules/setup/GeneratorMCDisc.hpp"
#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include <mpi.h>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
inline std::shared_ptr<shammodels::gsph::modules::IGSPHSetupNode> shammodels::gsph::modules::
    GSPHSetup<Tvec, SPHKernel>::make_generator_disc_mc(
        Tscal part_mass,
        Tscal disc_mass,
        Tscal r_in,
        Tscal r_out,
        std::function<Tscal(Tscal)> sigma_profile,
        std::function<Tscal(Tscal)> H_profile,
        std::function<Tvec(Tvec)> vel_profile,
        std::function<Tscal(Tvec)> cs_profile,
        std::mt19937_64 eng,
        Tscal init_h_factor) {
    return std::shared_ptr<IGSPHSetupNode>(new gsph::modules::GeneratorMCDisc<Tvec, SPHKernel>(
        context,
        solver_config,
        part_mass,
        disc_mass,
        r_in,
        r_out,
        sigma_profile,
        H_profile,
        vel_profile,
        cs_profile,
        eng,
        init_h_factor));
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::GSPHSetup<Tvec, SPHKernel>::apply_setup(
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
        sched.for_each_local_patchdata([&](const Patch &p, PatchDataLayer &pdat) {
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

    time_setup.stop();
    if (shamcomm::world_rank() == 0) {
        logger::info_ln("SPH setup", "the setup took :", time_setup.elapsed_sec(), "s");
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

using namespace shammath;
template class shammodels::gsph::modules::GSPHSetup<f64_3, M4>;
template class shammodels::gsph::modules::GSPHSetup<f64_3, M6>;
template class shammodels::gsph::modules::GSPHSetup<f64_3, M8>;

template class shammodels::gsph::modules::GSPHSetup<f64_3, C2>;
template class shammodels::gsph::modules::GSPHSetup<f64_3, C4>;
template class shammodels::gsph::modules::GSPHSetup<f64_3, C6>;

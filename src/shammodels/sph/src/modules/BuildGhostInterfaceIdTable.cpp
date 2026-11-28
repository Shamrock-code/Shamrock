// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BuildGhostInterfaceIdTable.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/modules/BuildGhostInterfaceIdTable.hpp"
#include "shamrock/patch/Patch.hpp"
#include <unordered_set>
#include <map>
#include <string>
#include <vector>

template<class Tvec>
void shammodels::sph::modules::BuildGhostInterfaceIdTable<Tvec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    using namespace shamrock::patch;

    auto edges = get_edges();

    // inputs
    auto &positions       = edges.positions;
    auto &interface_infos = edges.interface_infos.values;

    // positions only holds non-empty patches, an interface whose sender is absent from it has no
    // particles to send and is therefore dropped (as an empty interface would be). Since the filter
    // only holds **local** non-empty patches, an interface whose sender is not a local patch would
    // also be silently dropped here rather than raising an error.
    std::vector<u64> ids_vec = positions.get_refs().get_ids();
    std::unordered_set<u64> non_empty_senders(ids_vec.begin(), ids_vec.end());

    shambase::DistributedDataShared<InterfaceIdTable> res;

    std::map<u64, f64> send_count_stats;

    interface_infos.for_each([&](u64 sender, u64 receiver, const InterfaceBuildInfos &build) {
        if (non_empty_senders.find(sender) == non_empty_senders.end()) {
            return;
        }

        PatchDataField<Tvec> &xyz = positions.get_field(sender);

        sham::DeviceBuffer<u32> idxs_res = xyz.get_ids_where(
            [](auto access, u32 id, Tvec vmin, Tvec vmax) {
                return Patch::is_in_patch_converted(access[id], vmin, vmax);
            },
            build.cut_volume.lower,
            build.cut_volume.upper);

        u32 pcnt = idxs_res.get_size();

        // prevent sending empty patches
        if (pcnt == 0) {
            return;
        }

        f64 ratio = f64(pcnt) / f64(xyz.get_obj_cnt());

        shamlog_debug_ln(
            "InterfaceGen",
            "gen interface :",
            sender,
            "->",
            receiver,
            "volume ratio:",
            build.volume_ratio,
            "part_ratio:",
            ratio);

        res.add_obj(sender, receiver, InterfaceIdTable{build, std::move(idxs_res), ratio});

        send_count_stats[sender] += ratio;
    });

    bool has_warn = false;

    std::string warn_log = "";

    for (auto &[k, v] : send_count_stats) {
        if (v > 0.2) {
            has_warn = true;
        }
    }

    if (has_warn && shamcomm::world_rank() == 0) {
        warn_log = "\n    This can lead to high mpi "
                   "overhead, try to increase the patch split crit"
                   + warn_log;
    }

    if (has_warn) {
        logger::warn_ln("InterfaceGen", "High interface/patch volume ratio." + warn_log);
    }

    edges.interface_id_table.values = std::move(res);
}

template<class Tvec>
std::string shammodels::sph::modules::BuildGhostInterfaceIdTable<Tvec>::_impl_get_tex() const {
    return "TODO";
}

template class shammodels::sph::modules::BuildGhostInterfaceIdTable<f64_3>;

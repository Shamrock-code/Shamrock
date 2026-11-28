// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHGhostHandler.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief GSPH-specific ghost handler implementation
 */

#include "shambase/exception.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include <functional>
#include <vector>

using namespace shammodels::gsph;

template<class vec>
auto GSPHGhostHandler<vec>::gen_id_table_interfaces(GeneratorMap &&gen)
    -> shambase::DistributedDataShared<InterfaceIdTable> {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    shambase::DistributedDataShared<InterfaceIdTable> res;

    std::map<u64, f64> send_count_stats;

    const u32 ixyz = sched.pdl_old().template get_field_idx<vec>("xyz");

    gen.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchDataLayer &src = sched.patch_data.get_pdat(sender);
        PatchDataField<vec> &xyz             = src.get_field<vec>(ixyz);

        sham::DeviceBuffer<u32> idxs_res = xyz.get_ids_where(
            [](auto access, u32 id, vec vmin, vec vmax) {
                return Patch::is_in_patch_converted(access[id], vmin, vmax);
            },
            build.cut_volume.lower,
            build.cut_volume.upper);

        u32 pcnt = idxs_res.get_size();

        // prevent sending empty patches
        if (pcnt == 0) {
            return;
        }

        f64 ratio = f64(pcnt) / f64(src.get_obj_cnt());

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
        warn_log = "\n    High interf/patch volume. This can lead to high mpi "
                   "overhead, try to increase the patch split crit"
                   + warn_log;
    }

    if (has_warn) {
        logger::warn_ln("InterfaceGen", "High interface/patch volume ratio." + warn_log);
    }

    return res;
}

template<class vec>
void GSPHGhostHandler<vec>::gen_debug_patch_ghost(
    shambase::DistributedDataShared<InterfaceIdTable> &interf_info) {
    StackEntry stack_loc{};

    static u32 cnt_dump_debug = 0;

    std::string loc_graph = "";
    interf_info.for_each([&loc_graph](u64 send, u64 recv, InterfaceIdTable &info) {
        loc_graph += sham::format("    p{} -> p{}\n", send, recv);
    });

    sched.for_each_patch_data(
        [&](u64 id, shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            if (pdat.get_obj_cnt() > 0) {
                loc_graph += sham::format(
                    "    p{} [label= \"id={} N={}\"]\n", id, id, pdat.get_obj_cnt());
            }
        });

    std::string dot_graph = "";
    shamalgs::collective::gather_str(loc_graph, dot_graph);

    dot_graph = "strict digraph {\n" + dot_graph + "}";

    if (shamcomm::world_rank() == 0) {
        std::string fname = sham::format("ghost_graph_{}.dot", cnt_dump_debug);
        logger::info_ln("GSPH Ghost", "writing", fname);
        shambase::write_string_to_file(fname, dot_graph);
        cnt_dump_debug++;
    }
}

template class shammodels::gsph::GSPHGhostHandler<f64_3>;

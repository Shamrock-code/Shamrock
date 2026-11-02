// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BasicSPHGhosts.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/modules/BuildGhostInterfaceIdTable.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include <functional>
#include <vector>

using namespace shammodels::sph;

template<class vec>
auto BasicSPHGhostHandler<vec>::gen_id_table_interfaces(GeneratorMap &&gen)
    -> shambase::DistributedDataShared<InterfaceIdTable> {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    // ----------------------------------------------------------------------------------------
    // temporary wrapper to slowly migrate to the new solvergraph
    auto positions = std::make_shared<shamrock::solvergraph::FieldRefs<vec>>("", "");
    shamrock::solvergraph::DDPatchDataFieldRef<vec> positions_refs = {};
    sched.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        positions_refs.add_obj(p.id_patch, std::ref(pdat.get_field<vec>(0)));
    });
    positions->set_refs(positions_refs);

    auto interface_infos
        = std::make_shared<shamrock::solvergraph::DDSharedScalar<InterfaceBuildInfos>>("", "");
    interface_infos->values = std::forward<GeneratorMap>(gen);

    auto interface_id_table
        = std::make_shared<shamrock::solvergraph::DDSharedScalar<InterfaceIdTable>>("", "");

    modules::BuildGhostInterfaceIdTable<vec> node;
    node.set_edges(positions, interface_infos, interface_id_table);
    node.evaluate();
    // ----------------------------------------------------------------------------------------

    return std::move(interface_id_table->values);
}

template<class vec>
void BasicSPHGhostHandler<vec>::gen_debug_patch_ghost(
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
        logger::info_ln("SPH Ghost", "writing", fname);
        shambase::write_string_to_file(fname, dot_graph);
        cnt_dump_debug++;
    }
}

template class shammodels::sph::BasicSPHGhostHandler<f64_3>;

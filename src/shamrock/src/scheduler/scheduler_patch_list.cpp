// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file scheduler_patch_list.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include <random>
#include <vector>

void SchedulerPatchList::build_global() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    shamalgs::collective::vector_allgatherv(
        local, get_patch_mpi_type<3>(), global, get_patch_mpi_type<3>(), MPI_COMM_WORLD);
}

std::unordered_set<u64> SchedulerPatchList::build_local() {
    StackEntry stack_loc{};

    std::unordered_set<u64> out_ids;

    local.clear();
    for (const shamrock::patch::Patch &p : global) {
        // TODO add check node_owner_id valid
        if (i32(p.node_owner_id) == shamcomm::world_rank()) {
            local.push_back(p);
            out_ids.insert(p.id_patch);
        }
    }

    return out_ids;
}

void SchedulerPatchList::build_local_differantial(
    std::unordered_set<u64> &patch_id_lst,
    std::vector<u64> &to_send_idx,
    std::vector<u64> &to_recv_idx) {

    local.clear();

    for (u64 i = 0; i < global.size(); i++) {
        const shamrock::patch::Patch &p = global[i];

        bool was_owned = (patch_id_lst.find(p.id_patch) != patch_id_lst.end());

        // TODO add check node_owner_id valid
        if (i32(p.node_owner_id) == shamcomm::world_rank()) {
            local.push_back(p);

            if (!was_owned) {
                to_recv_idx.push_back(i);
                patch_id_lst.insert(p.id_patch);
            }
        } else {
            if (was_owned) {
                to_send_idx.push_back(i);
                patch_id_lst.erase(p.id_patch);
            }
        }
    }
}

void SchedulerPatchList::build_global_idx_map() {
    StackEntry stack_loc{};
    id_patch_to_global_idx.clear();

    u64 idx = 0;
    for (shamrock::patch::Patch p : global) {
        id_patch_to_global_idx[p.id_patch] = idx;
        idx++;
    }
}

void SchedulerPatchList::build_local_idx_map() {
    StackEntry stack_loc{};
    id_patch_to_local_idx.clear();

    u64 idx = 0;
    for (shamrock::patch::Patch p : local) {
        id_patch_to_local_idx[p.id_patch] = idx;
        idx++;
    }
}

void SchedulerPatchList::reset_local_pack_index() {
    StackEntry stack_loc{};
    for (shamrock::patch::Patch &p : local) {
        p.pack_node_index = u64_max;
    }
}

std::tuple<u64, u64, u64, u64, u64, u64, u64, u64> SchedulerPatchList::split_patch(u64 id_patch) {

    using namespace shamrock::patch;

    Patch &p0 = global[id_patch_to_global_idx[id_patch]];

    std::array<Patch, 8> splts = p0.get_split();

    p0 = splts[0]; // override existing patch

    splts[1].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[2].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[3].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[4].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[5].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[6].id_patch = _next_patch_id;
    _next_patch_id++;
    splts[7].id_patch = _next_patch_id;
    _next_patch_id++;

    // TODO use emplace_back instead
    u64 idx_p1 = global.size();
    global.push_back(splts[1]);

    u64 idx_p2 = idx_p1 + 1;
    global.push_back(splts[2]);

    u64 idx_p3 = idx_p2 + 1;
    global.push_back(splts[3]);

    u64 idx_p4 = idx_p3 + 1;
    global.push_back(splts[4]);

    u64 idx_p5 = idx_p4 + 1;
    global.push_back(splts[5]);

    u64 idx_p6 = idx_p5 + 1;
    global.push_back(splts[6]);

    u64 idx_p7 = idx_p6 + 1;
    global.push_back(splts[7]);

    return {
        id_patch_to_global_idx[id_patch], idx_p1, idx_p2, idx_p3, idx_p4, idx_p5, idx_p6, idx_p7};
}

void SchedulerPatchList::merge_patch(
    u64 idx0, u64 idx1, u64 idx2, u64 idx3, u64 idx4, u64 idx5, u64 idx6, u64 idx7) {

    using namespace shamrock::patch;

    Patch p = Patch::merge_patch(
        {global[idx0],
         global[idx1],
         global[idx2],
         global[idx3],
         global[idx4],
         global[idx5],
         global[idx6],
         global[idx7]});

    global[idx0] = p;
    global[idx1].set_err_mode();
    global[idx2].set_err_mode();
    global[idx3].set_err_mode();
    global[idx4].set_err_mode();
    global[idx5].set_err_mode();
    global[idx6].set_err_mode();
    global[idx7].set_err_mode();
}

namespace shamrock::patch {

    /**
     * @brief Serializes a Patch object to a JSON object.
     *
     * Note that this function is implemented here as it is the only place where we need it,
     * thus helping a bit with compile time
     *
     * @param j The JSON object to serialize to.
     * @param p The Patch object to serialize from.
     */
    inline void to_json(nlohmann::json &j, const Patch &p) {

        // u64 id_patch;
        // u64 pack_node_index;
        // u64 load_value;
        // std::array<u64,dim> coord_min;
        // std::array<u64,dim> coord_max;
        // u32 node_owner_id;

        j = nlohmann::json{
            {"id_patch", p.id_patch},
            {"pack_node_index", p.pack_node_index},
            {"load_value", p.load_value},
            {"coord_min", p.coord_min},
            {"coord_max", p.coord_max},
            {"node_owner_id", p.node_owner_id},
        };
    }

    /**
     * @brief Deserializes a JSON object to a Patch object.
     *
     * Note that this function is implemented here as it is the only place where we need it,
     * thus helping a bit with compile time
     *
     * @param j The JSON object to deserialize from.
     * @param p The Patch object to deserialize to.
     */
    inline void from_json(const nlohmann::json &j, Patch &p) {
        j.at("id_patch").get_to(p.id_patch);
        j.at("pack_node_index").get_to(p.pack_node_index);
        j.at("load_value").get_to(p.load_value);
        j.at("coord_min").get_to(p.coord_min);
        j.at("coord_max").get_to(p.coord_max);
        j.at("node_owner_id").get_to(p.node_owner_id);
    }

} // namespace shamrock::patch

void to_json(nlohmann::json &j, const SchedulerPatchList &p) {
    j = nlohmann::json{
        {"_next_patch_id", p._next_patch_id},
        {"global", p.global},
        //{"local", p.local}, // must be disabled to avoid differences between ranks
        {"is_load_values_up_to_date", p.is_load_values_up_to_date},
    };
}

void from_json(const nlohmann::json &j, SchedulerPatchList &p) {
    j.at("_next_patch_id").get_to(p._next_patch_id);
    j.at("global").get_to(p.global);
    // j.at("local").get_to(p.local); // must be disabled to avoid differences between ranks
    j.at("is_load_values_up_to_date").get_to(p.is_load_values_up_to_date);
}

// TODO move in a separate file
std::vector<shamrock::patch::Patch> make_fake_patch_list(u32 total_dtcnt, u64 div_limit) {

    using namespace shamrock::patch;

    std::vector<Patch> plist;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f32> split_val(0, 1);

    using namespace shamrock::scheduler;

    plist.push_back(Patch{
        0,
        u64_max,
        total_dtcnt,
        0,
        0,
        0,
        HilbertLoadBalance<u64>::max_box_sz,
        HilbertLoadBalance<u64>::max_box_sz,
        HilbertLoadBalance<u64>::max_box_sz,
        0,
    });

    bool listchanged = true;

    u64 id_cnt = 0;
    while (listchanged) {
        listchanged = false;

        std::vector<Patch> to_add;

        for (Patch &p : plist) {
            if (p.load_value > div_limit) {

                /*
                std::cout << "splitting : ( " <<
                    "[" << p.x_min << "," << p.x_max << "] " <<
                    "[" << p.y_min << "," << p.y_max << "] " <<
                    "[" << p.z_min << "," << p.z_max << "] " <<
                    " ) " << p.load_value <<  std::endl;
                    */

                u64 min_x = p.coord_min[0];
                u64 min_y = p.coord_min[1];
                u64 min_z = p.coord_min[2];

                u64 split_x = (((p.coord_max[0] - p.coord_min[0]) + 1) / 2) - 1 + min_x;
                u64 split_y = (((p.coord_max[1] - p.coord_min[1]) + 1) / 2) - 1 + min_y;
                u64 split_z = (((p.coord_max[2] - p.coord_min[2]) + 1) / 2) - 1 + min_z;

                u64 max_x = p.coord_max[0];
                u64 max_y = p.coord_max[1];
                u64 max_z = p.coord_max[2];

                u32 qte_m = split_val(eng) * p.load_value;
                u32 qte_p = p.load_value - qte_m;

                u32 qte_mm = split_val(eng) * qte_m;
                u32 qte_mp = qte_m - qte_mm;

                u32 qte_pm = split_val(eng) * qte_p;
                u32 qte_pp = qte_p - qte_pm;

                u32 qte_mmm = split_val(eng) * qte_mm;
                u32 qte_mmp = qte_mm - qte_mmm;

                u32 qte_mpm = split_val(eng) * qte_mp;
                u32 qte_mpp = qte_mp - qte_mpm;

                u32 qte_pmm = split_val(eng) * qte_pm;
                u32 qte_pmp = qte_pm - qte_pmm;

                u32 qte_ppm = split_val(eng) * qte_pp;
                u32 qte_ppp = qte_pp - qte_ppm;

                Patch child_mmm = Patch{
                    id_cnt,
                    u64_max,
                    qte_mmm,
                    min_x,
                    min_y,
                    min_z,
                    split_x,
                    split_y,
                    split_z,
                    0,
                };
                id_cnt++;

                Patch child_mmp = Patch{
                    id_cnt,
                    u64_max,
                    qte_mmp,
                    min_x,
                    min_y,
                    split_z + 1,
                    split_x,
                    split_y,
                    max_z,
                    0,
                };
                id_cnt++;

                Patch child_mpm = Patch{
                    id_cnt,
                    u64_max,
                    qte_mpm,
                    min_x,
                    split_y + 1,
                    min_z,
                    split_x,
                    max_y,
                    split_z,
                    0,
                };
                id_cnt++;

                Patch child_mpp = Patch{
                    id_cnt,
                    u64_max,
                    qte_mpp,
                    min_x,
                    split_y + 1,
                    split_z + 1,
                    split_x,
                    max_y,
                    max_z,
                    0,
                };
                id_cnt++;

                Patch child_pmm = Patch{
                    id_cnt,
                    u64_max,
                    qte_pmm,
                    split_x + 1,
                    min_y,
                    min_z,
                    max_x,
                    split_y,
                    split_z,
                    0,
                };
                id_cnt++;

                Patch child_pmp = Patch{
                    id_cnt,
                    u64_max,
                    qte_pmp,
                    split_x + 1,
                    min_y,
                    split_z + 1,
                    max_x,
                    split_y,
                    max_z,
                    0,
                };
                id_cnt++;

                Patch child_ppm = Patch{
                    id_cnt,
                    u64_max,
                    qte_ppm,
                    split_x + 1,
                    split_y + 1,
                    min_z,
                    max_x,
                    max_y,
                    split_z,
                    0,
                };
                id_cnt++;

                Patch child_ppp = Patch{
                    id_cnt,
                    u64_max,
                    qte_ppp,
                    split_x + 1,
                    split_y + 1,
                    split_z + 1,
                    max_x,
                    max_y,
                    max_z,
                    0,
                };
                id_cnt++;

                p = child_mmm;
                to_add.push_back(child_mmp);
                to_add.push_back(child_mpm);
                to_add.push_back(child_mpp);
                to_add.push_back(child_pmm);
                to_add.push_back(child_pmp);
                to_add.push_back(child_ppm);
                to_add.push_back(child_ppp);
            }
        }

        if (!to_add.empty()) {
            listchanged = true;

            plist.insert(plist.end(), to_add.begin(), to_add.end());
        }

        /*
        for(Patch & p : plist){
            std::cout << "( " <<
                "[" << p.x_min << "," << p.x_max << "] " <<
                "[" << p.y_min << "," << p.y_max << "] " <<
                "[" << p.z_min << "," << p.z_max << "] " <<
                " ) " << p.load_value << std::endl;
        }

        std::cout << "----- end cycle -----" << std::endl;
        */
    }

    return plist;
}

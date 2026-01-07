// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchScheduler.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <ctime>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

// TODO move types init out
void PatchScheduler::init_mpi_required_types() {

    // if(!patch::is_mpi_patch_type_active()){
    //     patch::create_MPI_patch_type();
    // }
}

void PatchScheduler::free_mpi_required_types() {

    // if(patch::is_mpi_patch_type_active()){
    //     patch::free_MPI_patch_type();
    // }
}

template<u32 dim>
void PatchScheduler::make_patch_base_grid(std::array<u32, dim> patch_count) {

    static_assert(dim == 3, "this is not implemented for dim != 3");

    u32 max_lin_patch_count = 0;
    for (u32 i = 0; i < dim; i++) {
        max_lin_patch_count = sycl::max(max_lin_patch_count, patch_count[i]);
    }

    u64 coord_div_fact = sham::roundup_pow2_clz(max_lin_patch_count);

    u64 sz_root_patch = PatchScheduler::max_axis_patch_coord_length / coord_div_fact;

    std::vector<shamrock::patch::PatchCoord<3>> coords;
    for (u32 x = 0; x < patch_count[0]; x++) {
        for (u32 y = 0; y < patch_count[1]; y++) {
            for (u32 z = 0; z < patch_count[2]; z++) {
                shamrock::patch::PatchCoord coord;

                coord.coord_min[0] = sz_root_patch * (x);
                coord.coord_min[1] = sz_root_patch * (y);
                coord.coord_min[2] = sz_root_patch * (z);
                coord.coord_max[0] = sz_root_patch * (x + 1) - 1;
                coord.coord_max[1] = sz_root_patch * (y + 1) - 1;
                coord.coord_max[2] = sz_root_patch * (z + 1) - 1;

                coords.push_back(coord);
            }
        }
    }

    shamrock::patch::PatchCoord bounds;
    bounds.coord_min[0] = 0;
    bounds.coord_min[1] = 0;
    bounds.coord_min[2] = 0;
    bounds.coord_max[0] = sz_root_patch * patch_count[0] - 1;
    bounds.coord_max[1] = sz_root_patch * patch_count[1] - 1;
    bounds.coord_max[2] = sz_root_patch * patch_count[2] - 1;

    get_sim_box().set_patch_coord_bounding_box(bounds);

    add_root_patches(coords);
}

template void PatchScheduler::make_patch_base_grid<3>(std::array<u32, 3> patch_count);

std::vector<u64> PatchScheduler::add_root_patches(
    std::vector<shamrock::patch::PatchCoord<3>> coords) {

    using namespace shamrock::patch;

    std::vector<u64> ret;

    for (auto coord : coords) {

        u32 node_owner_id = 0;

        Patch root;
        root.id_patch        = patch_list._next_patch_id;
        root.pack_node_index = u64_max;
        root.load_value      = 0;
        root.coord_min[0]    = coord.coord_min[0];
        root.coord_min[1]    = coord.coord_min[1];
        root.coord_min[2]    = coord.coord_min[2];
        root.coord_max[0]    = coord.coord_max[0];
        root.coord_max[1]    = coord.coord_max[1];
        root.coord_max[2]    = coord.coord_max[2];
        root.node_owner_id   = node_owner_id;

        patch_list.global.push_back(root);
        patch_list._next_patch_id++;

        if (shamcomm::world_rank() == node_owner_id) {
            patch_data.owned_data.add_obj(root.id_patch, PatchDataLayer(get_layout_ptr()));
            shamlog_debug_sycl_ln("Scheduler", "adding patch data");
        } else {
            shamlog_debug_sycl_ln(
                "Scheduler",
                "patch data wasn't added rank =",
                shamcomm::world_rank(),
                " ower =",
                node_owner_id);
        }

        patch_tree.insert_root_node(root.id_patch, coord);

        ret.push_back(root.id_patch);

        // auto [bmin,bmax] = get_sim_box().patch_coord_to_domain<u64_3>(root);
        //
        //
        // shamlog_debug_ln("Scheduler", "adding patch : [ (",
        // coord.x_min,
        // coord.y_min,
        // coord.z_min,") ] [ (",
        // coord.x_max,
        // coord.y_max,
        // coord.z_max,") ]", bmin,bmax
        //);
    }

    // build_local() is declared as nodiscard
    (void) patch_list.build_local();
    patch_list.reset_local_pack_index();
    patch_list.build_local_idx_map();
    patch_list.build_global_idx_map();

    patch_list.invalidate_load_values();

    return ret;
}

void PatchScheduler::allpush_data(shamrock::patch::PatchDataLayer &pdat) {

    shamlog_debug_ln("Scheduler", "pushing data obj cnt =", pdat.get_obj_cnt());

    for_each_patch_data([&](u64 id_patch,
                            shamrock::patch::Patch cur_p,
                            shamrock::patch::PatchDataLayer &pdat_sched) {
        auto variant_main = pdl().get_main_field_any();

        variant_main.visit([&](auto &arg) {
            using base_t = typename std::remove_reference<decltype(arg)>::type::field_T;

            if constexpr (shambase::VectorProperties<base_t>::dimension == 3) {
                auto [bmin, bmax] = get_sim_box().patch_coord_to_domain<base_t>(cur_p);

                shamlog_debug_sycl_ln(
                    "Scheduler", "pushing data in patch ", id_patch, "search range :", bmin, bmax);

                pdat_sched.insert_elements_in_range(pdat, bmin, bmax);
            } else {
                throw std::runtime_error("this does not yet work with dimension different from 3");
            }
        });
    });
}

void PatchScheduler::add_root_patch() {
    using namespace shamrock::patch;

    PatchCoord coord;
    coord.coord_min[0] = 0;
    coord.coord_min[1] = 0;
    coord.coord_min[2] = 0;
    coord.coord_max[0] = max_axis_patch_coord;
    coord.coord_max[1] = max_axis_patch_coord;
    coord.coord_max[2] = max_axis_patch_coord;

    add_root_patches({coord});

    patch_list.invalidate_load_values();
}

PatchScheduler::PatchScheduler(
    const shamrock::patch::PatchDataLayout pdl_ptr,
    u64 crit_split,
    u64 crit_merge)
    : pdl_ptr(pdl_ptr),
      patch_data(
          pdl_ptr.get_layer_layout(0), // @astodo should be a list
          {{0, 0, 0}, {max_axis_patch_coord, max_axis_patch_coord, max_axis_patch_coord}}) {

    crit_patch_split = crit_split;
    crit_patch_merge = crit_merge;
}

PatchScheduler::~PatchScheduler() {}

bool PatchScheduler::should_resize_box(bool node_in) {
    u16 tmp = node_in;
    u16 out = 0;
    shamcomm::mpi::Allreduce(&tmp, &out, 1, mpi_type_u16, MPI_MAX, MPI_COMM_WORLD);
    return out;
}

// TODO move Loadbalancing function to template state
void PatchScheduler::sync_build_LB(bool global_patch_sync, bool balance_load) {

    patch_list.check_load_values_valid();

    if (global_patch_sync)
        patch_list.build_global();

    if (balance_load) {
        // real load balancing
        shamrock::scheduler::LoadBalancingChangeList change_list
            = LoadBalancer::make_change_list(patch_list.global);

        // exchange data
        patch_data.apply_change_list(change_list, patch_list);
    }

    // rebuild local table
    owned_patch_id = patch_list.build_local();
}

template<>
std::tuple<f32_3, f32_3> PatchScheduler::get_box_tranform() {
    if (!pdl().check_main_field_type<f32_3>())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query single precision box the main field is not of f32_3 type");

    auto [bmin, bmax] = patch_data.sim_box.get_bounding_box<f32_3>();

    f32_3 translate_factor = bmin;
    f32_3 scale_factor     = (bmax - bmin) / LoadBalancer::max_box_sz;

    return {translate_factor, scale_factor};
}

template<>
std::tuple<f64_3, f64_3> PatchScheduler::get_box_tranform() {
    if (!pdl().check_main_field_type<f64_3>())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query single precision box the main field is not of f64_3 type");

    auto [bmin, bmax] = patch_data.sim_box.get_bounding_box<f64_3>();

    f64_3 translate_factor = bmin;
    f64_3 scale_factor     = (bmax - bmin) / LoadBalancer::max_box_sz;

    return {translate_factor, scale_factor};
}

template<>
std::tuple<f32_3, f32_3> PatchScheduler::get_box_volume() {
    if (!pdl().check_main_field_type<f32_3>())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query single precision box the main field is not of f32_3 type");

    return patch_data.sim_box.get_bounding_box<f32_3>();
}

template<>
std::tuple<f64_3, f64_3> PatchScheduler::get_box_volume() {
    if (!pdl().check_main_field_type<f64_3>())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query single precision box the main field is not of f64_3 type");

    return patch_data.sim_box.get_bounding_box<f64_3>();
}

template<>
std::tuple<i64_3, i64_3> PatchScheduler::get_box_volume() {
    if (!pdl().check_main_field_type<i64_3>())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query single precision box the main field is not of i64_3 type");

    return patch_data.sim_box.get_bounding_box<i64_3>();
}

// TODO clean the output of this function
void PatchScheduler::scheduler_step(bool do_split_merge, bool do_load_balancing) {
    StackEntry stack_loc{};

    // std::cout << dump_status();

    if (!is_mpi_sycl_interop_active())
        throw shambase::make_except_with_loc<std::runtime_error>(
            "sycl mpi interop not initialized");

    shambase::Timer timer;
    shamlog_debug_ln("Scheduler", "running scheduler step");

    struct SchedulerStepTimers {
        shambase::Timer global_timer;
        shambase::Timer metadata_sync;
        std::optional<shambase::Timer> global_idx_map_build    = {};
        std::optional<shambase::Timer> patch_tree_count_reduce = {};
        std::optional<shambase::Timer> gen_merge_split_rq      = {};
        std::optional<u32_2> split_merge_cnt                   = {};
        std::optional<shambase::Timer> apply_splits            = {};
        std::optional<shambase::Timer> load_balance_compute    = {};
        std::optional<u32> load_balance_move_op_cnt            = {};
        std::optional<shambase::Timer> load_balance_apply      = {};

        void print_stats() {
            if (shamcomm::world_rank() == 0) {
                f64 total       = global_timer.nanosec;
                std::string str = "";
                str += "Scheduler step timings : ";
                str += shambase::format(
                    "\n   metadata sync     : {:<10} ({:2.1f}%)",
                    metadata_sync.get_time_str(),
                    f64(100 * (metadata_sync.nanosec / total)));
                if (patch_tree_count_reduce) {
                    str += shambase::format(
                        "\n   patch tree reduce : {:<10} ({:2.1f}%)",
                        patch_tree_count_reduce->get_time_str(),
                        100 * (patch_tree_count_reduce->nanosec / total));
                }
                if (gen_merge_split_rq) {
                    str += shambase::format(
                        "\n   gen split merge   : {:<10} ({:2.1f}%)",
                        gen_merge_split_rq->get_time_str(),
                        100 * (gen_merge_split_rq->nanosec / total));
                }
                if (split_merge_cnt) {
                    str += shambase::format(
                        "\n   split / merge op  : {}/{}",
                        split_merge_cnt->x(),
                        split_merge_cnt->y());
                }
                if (apply_splits) {
                    str += shambase::format(
                        "\n   apply split merge : {:<10} ({:2.1f}%)",
                        apply_splits->get_time_str(),
                        100 * (apply_splits->nanosec / total));
                }
                if (load_balance_compute) {
                    str += shambase::format(
                        "\n   LB compute        : {:<10} ({:2.1f}%)",
                        load_balance_compute->get_time_str(),
                        100 * (load_balance_compute->nanosec / total));
                }
                if (load_balance_move_op_cnt) {
                    str += shambase::format(
                        "\n   LB move op cnt    : {}", *load_balance_move_op_cnt);
                }
                if (load_balance_apply) {
                    str += shambase::format(
                        "\n   LB apply          : {:<10} ({:2.1f}%)",
                        load_balance_apply->get_time_str(),
                        100 * (load_balance_apply->nanosec / total));
                }
                logger::info_ln("Scheduler", str);
            }
        }
    } timers;

    timers.global_timer.start();

    patch_list.check_load_values_valid();

    timers.metadata_sync.start();
    patch_list.build_global();
    timers.metadata_sync.end();

    // std::cout << dump_status();

    std::unordered_set<u64> split_rq;
    std::unordered_set<u64> merge_rq;

    if (do_split_merge) {
        // std::cout << dump_status() << std::endl;

        // std::cout << "build_global_idx_map" <<std::endl;
        timers.global_idx_map_build = shambase::Timer{};
        timers.global_idx_map_build->start(); // TODO check if it it used outside of split merge ->
                                              // maybe need to be put before the if
        patch_list.build_global_idx_map();
        timers.global_idx_map_build->end();

        // std::cout << dump_status() << std::endl;

        // std::cout << "tree partial_values_reduction" <<std::endl;
        timers.patch_tree_count_reduce = shambase::Timer{};
        timers.patch_tree_count_reduce->start();
        patch_tree.partial_values_reduction(patch_list.global, patch_list.id_patch_to_global_idx);
        timers.patch_tree_count_reduce->end();

        // std::cout << dump_status() << std::endl;

        // Generate merge and split request
        timers.gen_merge_split_rq = shambase::Timer{};
        timers.gen_merge_split_rq->start();
        split_rq = patch_tree.get_split_request(crit_patch_split);
        merge_rq = patch_tree.get_merge_request(crit_patch_merge);
        timers.gen_merge_split_rq->end();

        timers.split_merge_cnt = u32_2{split_rq.size(), merge_rq.size()};
        /*
        std::cout << "     |-> split rq : ";
        for(u64 i : split_rq){
            std::cout << i << " ";
        }std::cout << std::endl;
        //*/

        /*
        std::cout << "     |-> merge rq : ";
        for(u64 i : merge_rq){
            std::cout << i << " ";
        }std::cout << std::endl;
        //*/

        // std::cout << dump_status() << std::endl;

        // std::cout << "split_patches" <<std::endl;
        timers.apply_splits = shambase::Timer{};
        timers.apply_splits->start();
        split_patches(split_rq);
        timers.apply_splits->end();

        // std::cout << dump_status() << std::endl;

        // check not necessary if no splits
        patch_list.build_global_idx_map();

        set_patch_pack_values(merge_rq);
    }

    if (do_load_balancing) {
        StackEntry stack_loc{};
        timers.load_balance_compute = shambase::Timer{};
        timers.load_balance_compute->start();
        // generate LB change list
        shamrock::scheduler::LoadBalancingChangeList change_list
            = LoadBalancer::make_change_list(patch_list.global);
        timers.load_balance_compute->end();

        timers.load_balance_move_op_cnt = change_list.change_ops.size();

        timers.load_balance_apply = shambase::Timer{};
        timers.load_balance_apply->start();
        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
        timers.load_balance_apply->end();
    }

    // std::cout << dump_status();

    if (do_split_merge) {
        patch_list.build_local_idx_map();
        merge_patches(merge_rq);
    }

    // TODO should be moved out of the scheduler step
    owned_patch_id = patch_list.build_local();
    patch_list.reset_local_pack_index();
    patch_list.build_local_idx_map();
    patch_list.build_global_idx_map(); // TODO check if required : added because possible bug
                                       // because of for each patch & serial patch tree
    // update_local_dtcnt_value();
    // update_local_load_value(); disable the load value compute it should be done only in the
    // models

    if (split_rq.size() > 0 || merge_rq.size() > 0) {
        patch_list.invalidate_load_values();
    }

    // std::cout << dump_status();

    timers.global_timer.end();
    timers.print_stats();
}

/*
void SchedulerMPI::scheduler_step(bool do_split_merge,bool do_load_balancing){

    // update patch list
    patch_list.sync_global();


    if(do_split_merge){
        // rebuild patch index map
        patch_list.build_global_idx_map();

        // apply reduction on leafs and corresponding parents
        patch_tree.partial_values_reduction(
            patch_list.global,
            patch_list.id_patch_to_global_idx);

        // Generate merge and split request
        std::unordered_set<u64> split_rq = patch_tree.get_split_request(crit_patch_split);
        std::unordered_set<u64> merge_rq = patch_tree.get_merge_request(crit_patch_merge);


        // apply split requests
        // update patch_list.global same on every node
        // and split patchdata accordingly if owned
        // & update tree
        split_patches(split_rq);

        // update packing index
        // same operation on evey cluster nodes
        set_patch_pack_values(merge_rq);

        // update patch list
        // necessary to update load values in splitted patches
        // alternative : disable this step and set fake load values (load parent / 8)
        //alternative impossible if gravity because we have to compute the multipole
        owned_patch_id = patch_list.build_local();
        patch_list.sync_global();
    }

    if(do_load_balancing){
        // generate LB change list
        std::vector<std::tuple<u64, i32, i32,i32>> change_list =
            make_change_list(patch_list.global);

        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
    }

    if(do_split_merge){
        // apply merge requests
        // & update tree
        merge_patches(merge_rq);



        // if(Merge) update patch list
        if(! merge_rq.empty()){
            owned_patch_id = patch_list.build_local();
            patch_list.sync_global();
        }
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}
//*/

std::string PatchScheduler::dump_status() {

    using namespace shamrock::patch;

    std::stringstream ss;

    ss << "----- MPI Scheduler dump -----\n\n";
    ss << " -> SchedulerPatchList\n";

    ss << "    len global : " << patch_list.global.size() << "\n";
    ss << "    len local  : " << patch_list.local.size() << "\n";

    ss << "    global content : \n";
    for (Patch &p : patch_list.global) {

        ss << "      -> " << p.id_patch << " : " << p.load_value << " " << p.node_owner_id << " "
           << p.pack_node_index << " "
           << "( [" << p.coord_min[0] << "," << p.coord_max[0] << "] "
           << " [" << p.coord_min[1] << "," << p.coord_max[1] << "] "
           << " [" << p.coord_min[2] << "," << p.coord_max[2] << "] )\n";
    }
    ss << "    local content : \n";
    for (Patch &p : patch_list.local) {

        ss << "      -> id : " << p.id_patch << " : " << p.load_value << " " << p.node_owner_id
           << " " << p.pack_node_index << " "
           << "( [" << p.coord_min[0] << "," << p.coord_max[0] << "] "
           << " [" << p.coord_min[1] << "," << p.coord_max[1] << "] "
           << " [" << p.coord_min[2] << "," << p.coord_max[2] << "] )\n";
    }

    ss << shambase::format(
        "patch_list.id_patch_to_global_idx :\n{}\n", patch_list.id_patch_to_global_idx);
    ss << shambase::format(
        "patch_list.id_patch_to_local_idx :\n{}\n", patch_list.id_patch_to_local_idx);

    ss << " -> SchedulerPatchData\n";
    ss << "    owned data : \n";

    patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
        ss << "patch id : " << patch_id << " len = " << pdat.get_obj_cnt() << "\n";
    });

    /*
    for(auto & [k,pdat] : patch_data.owned_data){
        ss << "      -> id : " << k << " len : (" <<
            pdat.pos_s.size() << " " <<pdat.pos_d.size() << " " <<
            pdat.U1_s.size() << " " <<pdat.U1_d.size() << " " <<
            pdat.U3_s.size() << " " <<pdat.U3_d.size() << " "
        << ")\n";
    }
    */

    ss << " -> SchedulerPatchTree\n";

    for (auto &[k, pnode] : patch_tree.tree) {
        ss << shambase::format(
            "      -> id : {} -> ({}) <=> {} [{}, {}] (cl={} il={} l={} pid={})\n",
            k,
            pnode.tree_node.childs_nid,
            pnode.linked_patchid,
            pnode.patch_coord.coord_min,
            pnode.patch_coord.coord_max,
            pnode.tree_node.child_are_all_leafs,
            pnode.tree_node.is_leaf,
            pnode.tree_node.level,
            pnode.tree_node.parent_nid);
    }

    return ss.str();
}

std::string PatchScheduler::format_patch_coord(shamrock::patch::Patch p) {
    std::string ret;
    if (pdl().check_main_field_type<f32_3>()) {
        auto [bmin, bmax] = patch_data.sim_box.patch_coord_to_domain<f32_3>(p);
        ret               = shambase::format("coord = {} {}", bmin, bmax);
    } else if (pdl().check_main_field_type<f64_3>()) {
        auto [bmin, bmax] = patch_data.sim_box.patch_coord_to_domain<f64_3>(p);
        ret               = shambase::format("coord = {} {}", bmin, bmax);
    } else if (pdl().check_main_field_type<u32_3>()) {
        auto [bmin, bmax] = patch_data.sim_box.patch_coord_to_domain<u32_3>(p);
        ret               = shambase::format("coord = {} {}", bmin, bmax);
    } else if (pdl().check_main_field_type<u64_3>()) {
        auto [bmin, bmax] = patch_data.sim_box.patch_coord_to_domain<u64_3>(p);
        ret               = shambase::format("coord = {} {}", bmin, bmax);
    } else {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "the main field does not match any");
    }
    return ret;
}

template<class vec>
void check_locality_t(PatchScheduler &sched) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    sched.for_each_patch_data([&](u64 pid, Patch p, shamrock::patch::PatchDataLayer &pdat) {
        PatchDataField<vec> &main_field = pdat.get_field<vec>(0);
        auto [bmin_p0, bmax_p0]         = sched.patch_data.sim_box.patch_coord_to_domain<vec>(p);

        main_field.check_err_range(
            [&](vec val, vec vmin, vec vmax) {
                return Patch::is_in_patch_converted(val, vmin, vmax);
            },
            bmin_p0,
            bmax_p0,
            shambase::format("patch id = {}", pid));
    });
}

void PatchScheduler::check_patchdata_locality_corectness() {

    StackEntry stack_loc{};

    if (pdl().check_main_field_type<f32_3>()) {
        check_locality_t<f32_3>(*this);
    } else if (pdl().check_main_field_type<f64_3>()) {
        check_locality_t<f64_3>(*this);
    } else if (pdl().check_main_field_type<u32_3>()) {
        check_locality_t<u32_3>(*this);
    } else if (pdl().check_main_field_type<u64_3>()) {
        check_locality_t<u64_3>(*this);
    } else if (pdl().check_main_field_type<i64_3>()) {
        check_locality_t<i64_3>(*this);
    } else {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "the main field does not match any");
    }
}

void PatchScheduler::split_patches(std::unordered_set<u64> split_rq) {
    StackEntry stack_loc{};
    for (u64 tree_id : split_rq) {

        patch_tree.split_node(tree_id);
        PatchTree::Node &splitted_node = patch_tree.tree[tree_id];

        shamrock::patch::Patch old_patch
            = patch_list.global[patch_list.id_patch_to_global_idx[splitted_node.linked_patchid]];

        auto [idx_p0, idx_p1, idx_p2, idx_p3, idx_p4, idx_p5, idx_p6, idx_p7]
            = patch_list.split_patch(splitted_node.linked_patchid);

        u64 old_patch_id = splitted_node.linked_patchid;

        splitted_node.linked_patchid = u64_max;
        patch_tree.tree[splitted_node.tree_node.childs_nid[0]].linked_patchid
            = patch_list.global[idx_p0].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[1]].linked_patchid
            = patch_list.global[idx_p1].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[2]].linked_patchid
            = patch_list.global[idx_p2].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[3]].linked_patchid
            = patch_list.global[idx_p3].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[4]].linked_patchid
            = patch_list.global[idx_p4].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[5]].linked_patchid
            = patch_list.global[idx_p5].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[6]].linked_patchid
            = patch_list.global[idx_p6].id_patch;
        patch_tree.tree[splitted_node.tree_node.childs_nid[7]].linked_patchid
            = patch_list.global[idx_p7].id_patch;

        try {
            patch_data.split_patchdata(
                old_patch_id,
                {patch_list.global[idx_p0],
                 patch_list.global[idx_p1],
                 patch_list.global[idx_p2],
                 patch_list.global[idx_p3],
                 patch_list.global[idx_p4],
                 patch_list.global[idx_p5],
                 patch_list.global[idx_p6],
                 patch_list.global[idx_p7]});
        } catch (const PatchDataRangeCheckError &e) {
            logger::err_ln("SchedulerPatchData", "catched range issue with patchdata split");

            logger::raw_ln("   old patch", old_patch.id_patch, format_patch_coord(old_patch));

            logger::err_ln("Scheduler", "global patch list :");
            for (shamrock::patch::Patch &p : patch_list.global) {
                logger::raw_ln("   patch", p.id_patch, format_patch_coord(p));
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "\n Initial error : "
                + shambase::increase_indent(std::string("\n") + e.what(), "\n   |"));
        }
    }
}

inline void PatchScheduler::merge_patches(std::unordered_set<u64> merge_rq) {
    StackEntry stack_loc{};
    for (u64 tree_id : merge_rq) {

        PatchTree::Node &to_merge_node = patch_tree.tree[tree_id];

        // std::cout << "merging patch tree id : " << tree_id << "\n";

        u64 patch_id0 = patch_tree.tree[to_merge_node.tree_node.childs_nid[0]].linked_patchid;
        u64 patch_id1 = patch_tree.tree[to_merge_node.tree_node.childs_nid[1]].linked_patchid;
        u64 patch_id2 = patch_tree.tree[to_merge_node.tree_node.childs_nid[2]].linked_patchid;
        u64 patch_id3 = patch_tree.tree[to_merge_node.tree_node.childs_nid[3]].linked_patchid;
        u64 patch_id4 = patch_tree.tree[to_merge_node.tree_node.childs_nid[4]].linked_patchid;
        u64 patch_id5 = patch_tree.tree[to_merge_node.tree_node.childs_nid[5]].linked_patchid;
        u64 patch_id6 = patch_tree.tree[to_merge_node.tree_node.childs_nid[6]].linked_patchid;
        u64 patch_id7 = patch_tree.tree[to_merge_node.tree_node.childs_nid[7]].linked_patchid;

        // print list of patch that will merge
        // std::cout << format("  -> (%d %d %d %d %d %d %d %d)\n", patch_id0, patch_id1, patch_id2,
        // patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);

        if (patch_list.global[patch_list.id_patch_to_global_idx[patch_id0]].node_owner_id
            == shamcomm::world_rank()) {
            patch_data.merge_patchdata(
                patch_id0,
                {patch_id0,
                 patch_id1,
                 patch_id2,
                 patch_id3,
                 patch_id4,
                 patch_id5,
                 patch_id6,
                 patch_id7});
        }

        patch_list.merge_patch(
            patch_list.id_patch_to_global_idx[patch_id0],
            patch_list.id_patch_to_global_idx[patch_id1],
            patch_list.id_patch_to_global_idx[patch_id2],
            patch_list.id_patch_to_global_idx[patch_id3],
            patch_list.id_patch_to_global_idx[patch_id4],
            patch_list.id_patch_to_global_idx[patch_id5],
            patch_list.id_patch_to_global_idx[patch_id6],
            patch_list.id_patch_to_global_idx[patch_id7]);

        patch_tree.merge_node_dm1(tree_id);

        to_merge_node.linked_patchid = patch_id0;
    }
}

inline void PatchScheduler::set_patch_pack_values(std::unordered_set<u64> merge_rq) {

    for (u64 tree_id : merge_rq) {

        PatchTree::Node &to_merge_node = patch_tree.tree[tree_id];

        u64 idx_pack
            = patch_list.id_patch_to_global_idx[patch_tree.tree[to_merge_node.get_child_nid(0)]
                                                    .linked_patchid];

        // std::cout << "node id : " << patch_list.global[idx_pack].id_patch << " should merge with
        // : ";

        for (u8 i = 1; i < 8; i++) {
            // std::cout <<  patch_tree.tree[to_merge_node.get_child_nid(i)].linked_patchid << " ";
            patch_list
                .global[patch_list.id_patch_to_global_idx
                            [patch_tree.tree[to_merge_node.get_child_nid(i)].linked_patchid]]
                .pack_node_index
                = idx_pack;
        } // std::cout << std::endl;
    }
}

void PatchScheduler::dump_local_patches(std::string filename) {

    using namespace shamrock::patch;

    std::ofstream fout(filename);

    if (pdl().check_main_field_type<f32_3>()) {

        std::tuple<f32_3, f32_3> box_transform = get_box_tranform<f32_3>();

        for (const Patch &p : patch_list.local) {

            f32_3 box_min
                = f32_3{p.coord_min[0], p.coord_min[1], p.coord_min[2]} * std::get<1>(box_transform)
                  + std::get<0>(box_transform);
            f32_3 box_max = (f32_3{p.coord_max[0], p.coord_max[1], p.coord_max[2]} + 1)
                                * std::get<1>(box_transform)
                            + std::get<0>(box_transform);

            fout << p.id_patch << "|" << p.load_value << "|" << p.node_owner_id << "|"
                 << p.pack_node_index << "|" << box_min.x() << "|" << box_max.x() << "|"
                 << box_min.y() << "|" << box_max.y() << "|" << box_min.z() << "|" << box_max.z()
                 << "|" << "\n";
        }

        fout.close();

    } else if (pdl().check_main_field_type<f64_3>()) {

        std::tuple<f64_3, f64_3> box_transform = get_box_tranform<f64_3>();

        for (const Patch &p : patch_list.local) {

            f64_3 box_min
                = f64_3{p.coord_min[0], p.coord_min[1], p.coord_min[2]} * std::get<1>(box_transform)
                  + std::get<0>(box_transform);
            f64_3 box_max = (f64_3{p.coord_max[0], p.coord_max[1], p.coord_max[3]} + 1)
                                * std::get<1>(box_transform)
                            + std::get<0>(box_transform);

            fout << p.id_patch << "|" << p.load_value << "|" << p.node_owner_id << "|"
                 << p.pack_node_index << "|" << box_min.x() << "|" << box_max.x() << "|"
                 << box_min.y() << "|" << box_max.y() << "|" << box_min.z() << "|" << box_max.z()
                 << "|" << "\n";
        }

        fout.close();

    } else {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "the chosen type for the main field is not handled");
    }
}

struct Message {
    std::unique_ptr<shamcomm::CommunicationBuffer> buf;
    i32 rank;
    i32 tag;
};

void send_messages(std::vector<Message> &msgs, std::vector<MPI_Request> &rqs) {
    for (auto &msg : msgs) {
        rqs.push_back(MPI_Request{});
        u32 rq_index = rqs.size() - 1;
        auto &rq     = rqs[rq_index];

        u64 bsize = msg.buf->get_size();
        if (bsize % 8 != 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "the following mpi comm assume that we can send longs to pack 8byte");
        }
        u64 lcount = bsize / 8;
        if (lcount > i32_max) {
            shambase::throw_with_loc<std::runtime_error>("The message is too large for MPI");
        }

        shamcomm::mpi::Isend(
            msg.buf->get_ptr(),
            lcount,
            get_mpi_type<u64>(),
            msg.rank,
            msg.tag,
            MPI_COMM_WORLD,
            &rq);
    }
}

void recv_probe_messages(std::vector<Message> &msgs, std::vector<MPI_Request> &rqs) {

    for (auto &msg : msgs) {
        rqs.push_back(MPI_Request{});
        u32 rq_index = rqs.size() - 1;
        auto &rq     = rqs[rq_index];

        MPI_Status st;
        i32 cnt;
        shamcomm::mpi::Probe(msg.rank, msg.tag, MPI_COMM_WORLD, &st);
        shamcomm::mpi::Get_count(&st, get_mpi_type<u64>(), &cnt);

        msg.buf = std::make_unique<shamcomm::CommunicationBuffer>(
            cnt * 8, shamsys::instance::get_compute_scheduler_ptr());

        shamcomm::mpi::Irecv(
            msg.buf->get_ptr(), cnt, get_mpi_type<u64>(), msg.rank, msg.tag, MPI_COMM_WORLD, &rq);
    }
}

std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> PatchScheduler::gather_data(
    u32 rank) {

    using namespace shamrock::patch;

    auto plist = this->patch_list.global;
    auto pdata = this->patch_data.owned_data;

    auto serializer = [](shamrock::patch::PatchDataLayer &pdat) {
        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        ser.allocate(pdat.serialize_buf_byte_size());
        pdat.serialize_buf(ser);
        return ser.finalize();
    };

    auto deserializer = [&](sham::DeviceBuffer<u8> &&buf) {
        // exchange the buffer held by the distrib data and give it to the serializer
        shamalgs::SerializeHelper ser(
            shamsys::instance::get_compute_scheduler_ptr(),
            std::forward<sham::DeviceBuffer<u8>>(buf));
        return shamrock::patch::PatchDataLayer::deserialize_buf(ser, get_layout_ptr());
    };

    std::vector<Message> send_payloads;

    for (u32 i = 0; i < plist.size(); i++) {
        auto &cpatch = plist[i];
        if (cpatch.node_owner_id == shamcomm::world_rank()) {
            auto &patchdata = pdata.get(cpatch.id_patch);

            sham::DeviceBuffer<u8> tmp = serializer(patchdata);

            send_payloads.push_back(
                Message{
                    std::make_unique<shamcomm::CommunicationBuffer>(
                        std::move(tmp), shamsys::instance::get_compute_scheduler_ptr()),
                    0,
                    i32(i)});
        }
    }

    std::vector<MPI_Request> rqs;
    send_messages(send_payloads, rqs);

    std::vector<Message> recv_payloads;

    if (shamcomm::world_rank() == 0) {
        for (u32 i = 0; i < plist.size(); i++) {
            recv_payloads.push_back(
                Message{
                    std::unique_ptr<shamcomm::CommunicationBuffer>{},
                    i32(plist[i].node_owner_id),
                    i32(i)});
        }
    }

    // receive
    recv_probe_messages(recv_payloads, rqs);

    std::vector<MPI_Status> st_lst(rqs.size());
    shamcomm::mpi::Waitall(rqs.size(), rqs.data(), st_lst.data());

    std::vector<std::unique_ptr<PatchDataLayer>> ret;
    for (auto &recv_msg : recv_payloads) {
        shamcomm::CommunicationBuffer comm_buf = shambase::extract_pointer(recv_msg.buf);

        sham::DeviceBuffer<u8> buf
            = shamcomm::CommunicationBuffer::convert_usm(std::move(comm_buf));

        ret.push_back(std::make_unique<PatchDataLayer>(deserializer(std::move(buf))));
    }

    return ret;
}

nlohmann::json PatchScheduler::serialize_patch_metadata() {

    nlohmann::json jsim_box;
    patch_data.sim_box.to_json(jsim_box);

    return {
        {"patchtree", patch_tree},
        {"patchlist", patch_list},
        {"patchdata_layout", pdl()},
        {"sim_box", jsim_box},
        {"crit_patch_split", crit_patch_split},
        {"crit_patch_merge", crit_patch_merge}};
}

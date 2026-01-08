// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchScheduler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief MPI scheduler
 * @version 0.1
 * @date 2022-03-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamrock/legacy/patch/utility/patch_field.hpp"
#include "shamrock/solvergraph/NodeSetEdge.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
// #include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/Patch.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "scheduler_patch_list.hpp"
#include "shambackends/math.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamrock/scheduler/SchedulerPatchData.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

/**
 * @brief The MPI scheduler
 *
 */
class PatchScheduler {

    using LoadBalancer = shamrock::scheduler::HilbertLoadBalance<u64>;

    public:
    static constexpr u64 max_axis_patch_coord        = LoadBalancer::max_box_sz;
    static constexpr u64 max_axis_patch_coord_length = LoadBalancer::max_box_sz + 1;

    using PatchTree          = shamrock::scheduler::PatchTree;
    using SchedulerPatchData = shamrock::scheduler::SchedulerPatchData;

    shamrock::patch::PatchDataLayout pdl_ptr;

    u64 crit_patch_split; ///< splitting limit (if load value > crit_patch_split => patch split)
    u64 crit_patch_merge; ///< merging limit (if load value < crit_patch_merge => patch merge)

    SchedulerPatchList patch_list; ///< handle the list of the patches of the scheduler
    SchedulerPatchData patch_data; ///< handle the data of the patches of the scheduler
    PatchTree patch_tree;          ///< handle the tree structure of the patches

    // using unordered set is not an issue since we use the find command after
    std::unordered_set<u64> owned_patch_id; ///< list of owned patch ids updated with
    ///< (owned_patch_id = patch_list.build_local())

    inline shamrock::patch::PatchDataLayerLayout &pdl(size_t layer_idx = 0) {
        return pdl_ptr.get_layer_layout_ptr(layer_idx);
    }

    inline std::shared_ptr<shamrock::patch::PatchDataLayerLayout> get_layout_ptr(
        size_t layer_idx = 0) const {
        return pdl_ptr.get_layer_layout(layer_idx);
    }

    /**
     * @brief scheduler step
     *
     * @param do_split_merge
     * @param do_load_balancing
     */
    void scheduler_step(bool do_split_merge, bool do_load_balancing);

    void init_mpi_required_types();

    void free_mpi_required_types();

    PatchScheduler(const shamrock::patch::PatchDataLayout pdl_ptr, u64 crit_split, u64 crit_merge);

    ~PatchScheduler();

    std::string dump_status();

    inline void update_local_load_value(std::function<u64(shamrock::patch::Patch)> load_function) {
        for (u64 id : owned_patch_id) {
            shamrock::patch::Patch &p = patch_list.local[patch_list.id_patch_to_local_idx[id]];
            p.load_value              = load_function(p);
        }
        patch_list.is_load_values_up_to_date = true;
    }

    template<class vectype>
    std::tuple<vectype, vectype> get_box_tranform();

    template<class vectype>
    std::tuple<vectype, vectype> get_box_volume();

    bool should_resize_box(bool node_in);

    /**
     * @brief modify the bounding box of the patch domain
     *
     * @tparam vectype
     * @param bmin
     * @param bmax
     */
    template<class vectype>
    void set_coord_domain_bound(vectype bmin, vectype bmax) {

        size_t layer_idx = 0;
        for (const auto &layer_layout : pdl_ptr.layer_layouts) {
            if (!layer_layout->check_main_field_type<vectype>()) {
                throw std::invalid_argument(
                    std::string("the main field is not of the correct type to call this function\n")
                    + "fct called : " + __PRETTY_FUNCTION__
                    + shambase::format("current patch data layout of index {} : ", layer_idx)
                    + layer_layout->get_description_str());
            }
            layer_idx++;
        }

        patch_data.sim_box.set_bounding_box<vectype>({bmin, bmax});

        shamlog_debug_ln("PatchScheduler", "box resized to :", bmin, bmax);
    }

    /**
     * @brief push data in the scheduler
     * The content of pdat as to be the same for each node
     *
     * @param pdat the data to push
     */
    void allpush_data(shamrock::patch::PatchDataLayer &pdat);

    template<u32 dim>
    void make_patch_base_grid(std::array<u32, dim> patch_count);

    /**
     * @brief modify the bounding box of the patch domain
     *
     * @tparam vectype
     * @param box
     */
    template<class vectype>
    void set_coord_domain_bound(std::tuple<vectype, vectype> box) {
        auto [a, b] = box;
        set_coord_domain_bound(a, b);
    }

    std::string format_patch_coord(shamrock::patch::Patch p);

    void check_patchdata_locality_corectness();

    [[deprecated]]
    void dump_local_patches(std::string filename);

    std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> gather_data(u32 rank);

    /**
     * @brief add patch to the scheduler
     *
     * //TODO find a better way to do this it is too error prone
     *
     * @param p
     * @param pdat
     */
    //[[deprecated]]
    // inline u64 add_patch(shamrock::patch::Patch p, shamrock::patch::PatchData && pdat){
    //    p.id_patch = patch_list._next_patch_id;
    //    patch_list._next_patch_id ++;
    //
    //    patch_list.global.push_back(p);
    //
    //    patch_data.owned_data.insert({p.id_patch , pdat});
    //
    //    return p.id_patch;
    //}

    void add_root_patch();

    [[deprecated]]
    void sync_build_LB(bool global_patch_sync, bool balance_load);

    template<class vec>
    inline shamrock::patch::PatchCoordTransform<vec> get_patch_transform() {
        return get_sim_box().template get_patch_transform<vec>();
    }

    // template<class vec>
    // inline SerialPatchTree<vec> make_serial_ptree(){
    //     return SerialPatchTree<vec>(patch_tree, get_patch_transform<vec>());
    // }

    /**
     * @brief for each macro for patchadata
     * exemple usage
     * @code{.cpp}
     * sched.for_each_patch_data(
     *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) {
     *          ....
     *     }
     * );
     * @endcode
     *
     * @tparam Function The functor that will be used
     * @param fct
     */
    template<class Function>
    inline void for_each_patch_data(Function &&fct) {

        patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
            shamrock::patch::Patch &cur_p
                = patch_list.global[patch_list.id_patch_to_global_idx[patch_id]];

            if (!cur_p.is_err_mode()) {
                fct(patch_id, cur_p, pdat);
            }
        });
    }

    template<class Function>
    inline void for_each_patch(Function &&fct) {

        patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
            shamrock::patch::Patch &cur_p
                = patch_list.global[patch_list.id_patch_to_global_idx[patch_id]];

            // TODO should feed the sycl queue to the lambda
            if (!cur_p.is_err_mode()) {
                fct(patch_id, cur_p);
            }
        });
    }

    inline void for_each_global_patch(std::function<void(const shamrock::patch::Patch)> fct) {
        for (shamrock::patch::Patch p : patch_list.global) {
            if (!p.is_err_mode()) {
                fct(p);
            }
        }
    }

    inline void for_each_local_patch(std::function<void(const shamrock::patch::Patch)> fct) {
        for (shamrock::patch::Patch p : patch_list.local) {
            if (!p.is_err_mode()) {
                fct(p);
            }
        }
    }

    inline void for_each_local_patchdata(
        std::function<void(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &)> fct) {
        for (shamrock::patch::Patch p : patch_list.local) {
            if (!p.is_err_mode()) {
                fct(p, patch_data.get_pdat(p.id_patch));
            }
        }
    }

    inline void for_each_local_patch_nonempty(
        std::function<void(const shamrock::patch::Patch &)> fct) {
        patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
            shamrock::patch::Patch &cur_p
                = patch_list.global[patch_list.id_patch_to_global_idx.at(patch_id)];

            if ((!cur_p.is_err_mode()) && (!pdat.is_empty())) {
                fct(cur_p);
            }
        });
    }

    inline u32 get_patch_rank_owner(u64 patch_id) {
        shamrock::patch::Patch &cur_p
            = patch_list.global[patch_list.id_patch_to_global_idx.at(patch_id)];
        return cur_p.node_owner_id;
    }

    inline void for_each_patchdata_nonempty(
        std::function<void(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &)> fct) {
        patch_data.for_each_patchdata([&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
            shamrock::patch::Patch &cur_p
                = patch_list.global[patch_list.id_patch_to_global_idx.at(patch_id)];

            if ((!cur_p.is_err_mode()) && (!pdat.is_empty())) {
                fct(cur_p, pdat);
            }
        });
    }

    template<class T>
    inline shambase::DistributedData<T> map_owned_patchdata(
        std::function<T(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> fct) {
        shambase::DistributedData<T> ret;

        using namespace shamrock::patch;
        for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            ret.add_obj(id_patch, fct(cur_p, pdat));
        });

        return ret;
    }

    template<class T>
    inline shambase::DistributedData<T> distrib_data_local_to_all_simple(
        shambase::DistributedData<T> &src) {
        using namespace shamrock::patch;

        // TODO : after a split the scheduler patch list state does not match global =
        // allgather(local) but here it is implicitely assumed, that's ... bad
        return shamalgs::collective::fetch_all_simple<T, Patch>(
            src, patch_list.local, patch_list.global, [](Patch p) {
                return p.id_patch;
            });
    }

    template<class T>
    inline shambase::DistributedData<T> distrib_data_local_to_all_load_store(
        shambase::DistributedData<T> &src) {
        using namespace shamrock::patch;

        return shamalgs::collective::fetch_all_storeload<T, Patch>(
            src, patch_list.local, patch_list.global, [](Patch p) {
                return p.id_patch;
            });
    }

    template<class T>
    inline shambase::DistributedData<T> map_owned_patchdata_fetch_simple(
        std::function<T(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> fct) {
        shambase::DistributedData<T> ret;

        using namespace shamrock::patch;
        for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            ret.add_obj(id_patch, fct(cur_p, pdat));
        });

        return distrib_data_local_to_all_simple(ret);
    }

    template<class T>
    inline shambase::DistributedData<T> map_owned_patchdata_fetch_load_store(
        std::function<T(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> fct) {
        shambase::DistributedData<T> ret;

        using namespace shamrock::patch;
        for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            ret.add_obj(id_patch, fct(cur_p, pdat));
        });

        return distrib_data_local_to_all_load_store(ret);
    }

    template<class T>
    inline shamrock::patch::PatchField<T> map_owned_to_patch_field_simple(
        std::function<T(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> fct) {
        return shamrock::patch::PatchField<T>(map_owned_patchdata_fetch_simple(fct));
    }

    template<class T>
    inline shamrock::patch::PatchField<T> map_owned_to_patch_field_load_store(
        std::function<T(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> fct) {
        return shamrock::patch::PatchField<T>(map_owned_patchdata_fetch_load_store(fct));
    }

    inline u64 get_rank_count() {
        StackEntry stack_loc{};
        using namespace shamrock::patch;
        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            num_obj += pdat.get_obj_cnt();
        });

        return num_obj;
    }

    inline u64 get_total_obj_count() {
        StackEntry stack_loc{};
        u64 part_cnt = get_rank_count();
        return shamalgs::collective::allreduce_sum(part_cnt);
    }

    template<class T>
    inline std::unique_ptr<sycl::buffer<T>> rankgather_field(u32 field_idx) {
        StackEntry stack_loc{};
        std::unique_ptr<sycl::buffer<T>> ret;

        auto fd  = pdl().get_field<T>(field_idx);
        u64 nvar = fd.nvar;

        u64 num_obj = get_rank_count();

        if (num_obj > 0) {
            ret = std::make_unique<sycl::buffer<T>>(num_obj * nvar);

            using namespace shamrock::patch;

            u64 ptr = 0; // TODO accumulate_field() in scheduler ?
            for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
                using namespace shamalgs::memory;
                using namespace shambase;

                if (pdat.get_obj_cnt() > 0) {
                    write_with_offset_into(
                        shamsys::instance::get_compute_scheduler().get_queue(),
                        get_check_ref(ret),
                        pdat.get_field<T>(field_idx).get_buf(),
                        ptr,
                        pdat.get_obj_cnt() * nvar);

                    ptr += pdat.get_obj_cnt() * nvar;
                }
            });
        }

        return ret;
    }

    // template<class Function, class Pfield>
    // inline void compute_patch_field(Pfield & field, MPI_Datatype & dtype , Function && lambda){
    //     field.local_nodes_value.resize(patch_list.local.size());
    //
    //
    //
    //     for (u64 idx = 0; idx < patch_list.local.size(); idx++) {
    //
    //         Patch &cur_p = patch_list.local[idx];
    //
    //         PatchDataBuffer pdatbuf =
    //         attach_to_patchData(patch_data.owned_data.at(cur_p.id_patch));
    //
    //         field.local_nodes_value[idx] =
    //         lambda(shamsys::instance::get_compute_queue(),cur_p,pdatbuf);
    //
    //     }
    //
    //     field.build_global(dtype);
    //
    // }

    template<class Function, class Pfield>
    inline void compute_patch_field(Pfield &field, MPI_Datatype &dtype, Function &&lambda) {
        field.local_nodes_value.resize(patch_list.local.size());

        for (u64 idx = 0; idx < patch_list.local.size(); idx++) {

            shamrock::patch::Patch &cur_p = patch_list.local[idx];

            if (!cur_p.is_err_mode()) {
                field.local_nodes_value[idx] = lambda(
                    shamsys::instance::get_compute_queue(),
                    cur_p,
                    patch_data.owned_data.get(cur_p.id_patch));
            }
        }

        field.build_global(dtype);
    }

    inline auto get_node_set_edge_patchdata_layer_refs() {
        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::PatchDataLayerRefs> node_set_edge(
            [&](shamrock::solvergraph::PatchDataLayerRefs &edge) {
                edge.free_alloc();
                using namespace shamrock::patch;
                for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                    edge.patchdatas.add_obj(cur_p.id_patch, std::ref(pdat));
                });
            });

        return std::make_shared<decltype(node_set_edge)>(std::move(node_set_edge));
    };

    /**
     * @brief add a root patch to the scheduler
     *
     * @param coords coordinates of the patch
     * @return u64 the id of the made patch
     */
    std::vector<u64> add_root_patches(std::vector<shamrock::patch::PatchCoord<3>> coords);

    shamrock::patch::SimulationBoxInfo &get_sim_box() { return patch_data.sim_box; }

    nlohmann::json serialize_patch_metadata();

    private:
    void split_patches(std::unordered_set<u64> split_rq);
    void merge_patches(std::unordered_set<u64> merge_rq);

    void set_patch_pack_values(std::unordered_set<u64> merge_rq);
};

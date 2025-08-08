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
 * @file interface_handler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

//%Impl status : Clean unfinished
// move to "good" when new handler implemented

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/interfaces/interface_generator.hpp"
#include <memory>
#include <vector>
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "interface_handler_impl.hpp"
#include "interface_handler_impl/interface_handler_impl_list.hpp"
#include "interface_handler_impl/interface_handler_impl_tree.hpp"
#include "shamrock/legacy/patch/utility/sphpatch.hpp" // TODO remove sph dependancy
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

/**
 * @brief
 *
 * @tparam vectype
 * @tparam primtype
 * //TODO check that for periodic BC case : if a patch has itself as interface, is there any bug
 * because of map {map } repr
 * //TODO put the flag choice thing in a separate function to avoid recomputation
 */
template<class vectype, class primtype>
class LegacyInterfacehandler {

    private:
    /**
     * @brief
     *
     */
    std::vector<InterfaceComm<vectype>> interface_comm_list;

    /**
     * @brief
     *
     */
    std::unordered_map<
        u64,
        std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchDataLayer>>>>
        interface_map;

    public:
    /**
     * @brief
     *
     * @tparam interface_selector
     * @param sched
     * @param sptree
     * @param h_field
     */
    template<class interface_selector>
    inline void compute_interface_list(
        PatchScheduler &sched,
        SerialPatchTree<vectype> &sptree,
        legacy::PatchField<primtype> h_field,
        bool periodic) {
        StackEntry stack_loc{};
        interface_comm_list
            = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
                sched,
                sptree,
                h_field,
                shambase::format_printf("interfaces_%d_node%d", 0, shamcomm::world_rank()),
                periodic);
    }

    /**
     * @brief
     *
     * @param sched
     */
    void comm_interfaces(PatchScheduler &sched, bool periodic);

    template<class T>
    PatchComputeFieldInterfaces<T>
    comm_interfaces_field(PatchScheduler &sched, PatchComputeField<T> &pcomp_field, bool periodic) {
        StackEntry stack_loc{};
        PatchComputeFieldInterfaces<T> interface_field_map;

        impl::comm_interfaces_field<T, vectype>(
            sched, pcomp_field, interface_comm_list, interface_field_map.interface_map, periodic);

        return interface_field_map;
    }

    /**
     * @brief Get the interface list object
     *
     * @param key
     * @return const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>&
     */
    inline const std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchDataLayer>>> &
    get_interface_list(u64 key) {
        return interface_map[key];
    }

    /**
     * @brief
     *
     */
    inline void print_current_interf_map() {

        for (const auto &[pid, int_vec] : interface_map) {
            logger::raw_ln(shambase::format_printf(" pid : %lu :\n", pid));
            // for (auto &[a, b] : int_vec) {
            //     //printf("    -> %d : len %d\n", a, b->obj_cnt);
            // }
        }
    }

    // template<class Function>
    //[[deprecated]]
    // inline void for_each_interface_buf(u64 patch_id,sycl::queue & queue, Function && fct){
    //
    //    const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst =
    //    get_interface_list(patch_id);
    //
    //    for (auto & [int_pid, pdat_ptr] : p_interf_lst) {
    //
    //        if(! pdat_ptr->is_empty()){
    //
    //            PatchDataBuffer pdat_buf = attach_to_patchData(*pdat_ptr);
    //
    //            auto t = patchdata::sph::get_patchdata_BBAA<vectype>(queue, pdat_buf);
    //
    //            fct(patch_id,int_pid,pdat_buf,t);
    //        }
    //    }
    //
    //}

    template<class Function>
    inline void for_each_interface(u64 patch_id, Function &&fct) {

        using namespace shamrock::patch;

        const std::vector<std::tuple<u64, std::unique_ptr<PatchDataLayer>>> &p_interf_lst
            = get_interface_list(patch_id);

        for (auto &[int_pid, pdat_ptr] : p_interf_lst) {

            if (!pdat_ptr->is_empty()) {

                PatchDataLayer &pdat = *pdat_ptr;

                u32 ixyz = pdat.pdl().get_field_idx<vectype>("xyz");

                u32 nobj = pdat.get_obj_cnt();

                auto &buf = pdat.get_field<vectype>(ixyz).get_buf();
                auto t    = syclalg::get_min_max<vectype>(buf, nobj);

                fct(patch_id, int_pid, pdat, t);
            }
        }
    }
};

template class LegacyInterfacehandler<f32_3, f32>;
template class LegacyInterfacehandler<f64_3, f64>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file interface_generator.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/exchanges.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/Patch.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shambase/string.hpp"
#include "interface_generator_impl.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SchedulerPatchData.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <unordered_map>
#include <cstddef>
#include <fstream>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <vector>

class InterfaceVolumeGenerator {
    public:
    template<class vectype>
    static std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> append_interface(
        sycl::queue &queue,
        shamrock::patch::PatchDataLayer &pdat_buf,
        std::vector<vectype> boxs_min,
        std::vector<vectype> boxs_max,
        vectype add_offset);

    template<class T, class vectype>
    inline static std::vector<std::unique_ptr<PatchDataField<T>>> append_interface_field(
        sycl::queue &queue,
        shamrock::patch::PatchDataLayer &pdat_buf,
        PatchDataField<T> &pdat_cfield,
        std::vector<vectype> boxs_min,
        std::vector<vectype> boxs_max) {
        return impl::append_interface_field<T, vectype>(
            queue, pdat_buf, pdat_cfield, boxs_min, boxs_max);
    }
};

template<class vectype>
struct InterfaceComm {
    u64 global_patch_idx_send;
    u64 global_patch_idx_recv;
    u64 sender_patch_id;
    u64 receiver_patch_id;
    vectype interf_box_min;
    vectype interf_box_max;
    vectype interf_offset;
};

template<class vectype, class field_type, class InterfaceSelector>
class Interface_Generator {

    private:
    struct InterfaceComInternal {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;
        vectype interf_box_min;
        vectype interf_box_max;
        vectype interf_offset;
    };

    inline static sycl::buffer<InterfaceComInternal, 2> get_interface_list_v1(
        PatchScheduler &sched,
        SerialPatchTree<vectype> &sptree,
        legacy::PatchField<typename vectype::element_type> pfield,
        vectype interf_offset) {

        const u64 local_pcount  = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        if (local_pcount == 0)
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "local patch count is zero this function can not run");

        sycl::buffer<u64> patch_ids_buf(local_pcount);
        sycl::buffer<vectype> local_box_min_buf(local_pcount);
        sycl::buffer<vectype> local_box_max_buf(local_pcount);

        sycl::buffer<vectype> global_box_min_buf(global_pcount);
        sycl::buffer<vectype> global_box_max_buf(global_pcount);

        sycl::buffer<InterfaceComInternal, 2> interface_list_buf({local_pcount, global_pcount});
        sycl::buffer<u64> global_ids_buf(global_pcount);

        {
            sycl::host_accessor pid{patch_ids_buf, sycl::write_only, sycl::no_init};
            sycl::host_accessor lbox_min{local_box_min_buf, sycl::write_only, sycl::no_init};
            sycl::host_accessor lbox_max{local_box_max_buf, sycl::write_only, sycl::no_init};

            sycl::host_accessor gbox_min{global_box_min_buf, sycl::write_only, sycl::no_init};
            sycl::host_accessor gbox_max{global_box_max_buf, sycl::write_only, sycl::no_init};

            std::tuple<vectype, vectype> box_transform = sched.get_box_tranform<vectype>();

            for (u64 i = 0; i < local_pcount; i++) {
                pid[i] = sched.patch_list.local[i].id_patch;

                lbox_min[i] = vectype{sched.patch_list.local[i].coord_min[0], sched.patch_list.local[i].coord_min[1],
                                      sched.patch_list.local[i].coord_min[2]} *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform);
                lbox_max[i] = (vectype{
                                   sched.patch_list.local[i].coord_max[0],
                                   sched.patch_list.local[i].coord_max[1],
                                   sched.patch_list.local[i].coord_max[2]}
                               + 1)
                                  * std::get<1>(box_transform)
                              + std::get<0>(box_transform);
            }

            sycl::host_accessor g_pid{global_ids_buf, sycl::write_only, sycl::no_init};
            for (u64 i = 0; i < global_pcount; i++) {
                g_pid[i] = sched.patch_list.global[i].id_patch;

                gbox_min[i] = vectype{sched.patch_list.global[i].coord_min[0], sched.patch_list.global[i].coord_min[1],
                                      sched.patch_list.global[i].coord_min[2]} *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform)
                              + interf_offset; //to handle offset on interfaces
                gbox_max[i] = (vectype{
                                   sched.patch_list.global[i].coord_max[0],
                                   sched.patch_list.global[i].coord_max[1],
                                   sched.patch_list.global[i].coord_max[2]}
                               + 1)
                                  * std::get<1>(box_transform)
                              + std::get<0>(box_transform)
                              + interf_offset; // to handle offset on interfaces
            }
        }

        // PatchFieldReduction<field_type> pfield_reduced = sptree.template reduce_field<field_type,
        //  OctreeMaxReducer>(hndl.alt_queues[0], sched, pfield);

        sycl::buffer<field_type> buf_local_field_val(
            pfield.local_nodes_value.data(), pfield.local_nodes_value.size());
        sycl::buffer<field_type> buf_global_field_val(
            pfield.global_values.data(), pfield.global_values.size());

        shamsys::instance::get_alt_queue().submit([&](sycl::handler &cgh) {
            auto pid  = patch_ids_buf.get_access<sycl::access::mode::read>(cgh);
            auto gpid = global_ids_buf.get_access<sycl::access::mode::read>(cgh);

            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read>(cgh);
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read>(cgh);

            auto gbox_min = global_box_min_buf.template get_access<sycl::access::mode::read>(cgh);
            auto gbox_max = global_box_max_buf.template get_access<sycl::access::mode::read>(cgh);

            auto local_field
                = buf_local_field_val.template get_access<sycl::access::mode::read>(cgh);
            auto global_field
                = buf_global_field_val.template get_access<sycl::access::mode::read>(cgh);

            auto interface_list
                = interface_list_buf.template get_access<sycl::access::mode::discard_write>(cgh);

            u64 cnt_patch = global_pcount;

            vectype offset = -interf_offset;

            bool is_off_not_bull = (offset.x() == 0) && (offset.y() == 0) && (offset.z() == 0);

            cgh.parallel_for(sycl::range<1>(local_pcount), [=](sycl::item<1> item) {
                u64 cur_patch_idx    = (u64) item.get_id(0);
                u64 cur_patch_id     = pid[cur_patch_idx];
                vectype cur_lbox_min = lbox_min[cur_patch_idx];
                vectype cur_lbox_max = lbox_max[cur_patch_idx];

                u64 interface_ptr = 0;

                for (u64 test_patch_idx = 0; test_patch_idx < cnt_patch; test_patch_idx++) {

                    vectype test_lbox_min = gbox_min[test_patch_idx];
                    vectype test_lbox_max = gbox_max[test_patch_idx];
                    u64 test_patch_id     = gpid[test_patch_idx];

                    {
                        std::tuple<vectype, vectype> b1 = InterfaceSelector::get_neighbourg_box_sz(
                            cur_lbox_min,
                            cur_lbox_max,
                            global_field[test_patch_idx],
                            local_field[cur_patch_idx]);
                        std::tuple<vectype, vectype> b2 = InterfaceSelector::get_compute_box_sz(
                            test_lbox_min,
                            test_lbox_max,
                            global_field[test_patch_idx],
                            local_field[cur_patch_idx]);

                        bool int_cd = ((!is_off_not_bull) || (test_patch_id != cur_patch_id));

                        if (BBAA::intersect_not_null_cella_b(
                                std::get<0>(b1), std::get<1>(b1), std::get<0>(b2), std::get<1>(b2))
                            && (int_cd)) {

                            std::tuple<vectype, vectype> box_interf = BBAA::get_intersect_cella_b(
                                std::get<0>(b1), std::get<1>(b1), std::get<0>(b2), std::get<1>(b2));
                            interface_list[{cur_patch_idx, interface_ptr}] = InterfaceComInternal{
                                cur_patch_idx,
                                test_patch_idx,
                                cur_patch_id,
                                test_patch_id,
                                std::get<0>(box_interf),
                                std::get<1>(box_interf),
                                offset};
                            interface_ptr++;
                        }
                    }
                }

                if (interface_ptr < global_pcount) {
                    interface_list[{cur_patch_idx, interface_ptr}] = InterfaceComInternal{
                        u64_max, u64_max, u64_max, u64_max, vectype{}, vectype{}, vectype{}};
                }
            });
        });

        // // now the list of interface is known
        // {
        //     auto interface_list = interface_list_buf.get_access<sycl::access::mode::read>();

        //     for (u64 i = 0; i < local_pcount; i++) {
        //         std::cout << "- " << sched.patch_list.local[i].id_patch << " : ";
        //         for (u64 j = 0; j < global_pcount; j++) {
        //             if (interface_list[{i, j}].x() == u64_max)
        //                 break;
        //             std::cout << "(" << sched.patch_list.local[interface_list[{i,
        //             j}].x()].id_patch << ","
        //                       << sched.patch_list.global[interface_list[{i, j}].y()].id_patch <<
        //                       ") ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        return interface_list_buf;
    }

    public:
    /**
     * @brief
     *
     * @param sched
     * @param sptree
     * @param pfield the interaction radius field
     */
    inline static std::vector<InterfaceComm<vectype>> get_interfaces_comm_list(
        PatchScheduler &sched,
        SerialPatchTree<vectype> &sptree,
        legacy::PatchField<typename vectype::element_type> pfield,
        std::string fout,
        bool periodic) {

        const u64 local_pcount  = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        std::vector<InterfaceComm<vectype>> comm_vec;
        if (local_pcount != 0) {

            std::ofstream write_out(fout);

            auto add_interfaces = [&](vectype offset) {
                sycl::buffer<InterfaceComInternal, 2> interface_list_buf
                    = get_interface_list_v1(sched, sptree, pfield, offset);
                {
                    sycl::host_accessor interface_list{interface_list_buf, sycl::read_only};

                    for (u64 i = 0; i < local_pcount; i++) {
                        // std::cout << "- " << sched.patch_list.local[i].id_patch << " : ";
                        for (u64 j = 0; j < global_pcount; j++) {

                            // std::cout << "(" <<
                            // sched.patch_list.id_patch_to_global_idx[interface_list[{i,
                            // j}].sender_patch_id] << ","
                            //          << interface_list[{i, j}].global_patch_idx_recv << ") ";

                            if (interface_list[{i, j}].sender_patch_id == u64_max)
                                break;
                            // std::cout << "(" <<
                            // sched.patch_list.id_patch_to_global_idx[interface_list[{i,
                            // j}].sender_patch_id] << ","
                            //         << interface_list[{i, j}].global_patch_idx_recv << ") ";

                            InterfaceComInternal tmp = interface_list[{i, j}];

                            comm_vec.push_back(
                                InterfaceComm<vectype>{
                                    sched.patch_list.id_patch_to_global_idx[tmp.sender_patch_id],
                                    tmp.global_patch_idx_recv,
                                    tmp.sender_patch_id,
                                    tmp.receiver_patch_id,
                                    tmp.interf_box_min,
                                    tmp.interf_box_max,
                                    tmp.interf_offset});

                            write_out << interface_list[{i, j}].local_patch_idx_send << "|"
                                      << interface_list[{i, j}].global_patch_idx_recv << "|"
                                      << interface_list[{i, j}].sender_patch_id << "|"
                                      << interface_list[{i, j}].receiver_patch_id << "|"
                                      << interface_list[{i, j}].interf_box_min.x() << "|"
                                      << interface_list[{i, j}].interf_box_max.x() << "|"
                                      << interface_list[{i, j}].interf_box_min.y() << "|"
                                      << interface_list[{i, j}].interf_box_max.y() << "|"
                                      << interface_list[{i, j}].interf_box_min.z() << "|"
                                      << interface_list[{i, j}].interf_box_max.z() << "|"
                                      << interface_list[{i, j}].interf_offset.x() << "|"
                                      << interface_list[{i, j}].interf_offset.y() << "|"
                                      << interface_list[{i, j}].interf_offset.z() << "|" << "\n";
                        }
                        // std::cout << std::endl;
                    }
                }
            };

            if (!periodic) {
                add_interfaces({0, 0, 0});
            } else {
                auto [min_box, max_box] = sched.get_box_volume<vectype>();
                vectype offset_base     = max_box - min_box;

                for (i32 ix : {-1, 0, 1}) {
                    for (i32 iy : {-1, 0, 1}) {
                        for (i32 iz : {-1, 0, 1}) {
                            add_interfaces({
                                offset_base.x() * ix,
                                offset_base.y() * iy,
                                offset_base.z() * iz,
                            });
                        }
                    }
                }
            }

            write_out.close();
        }

        return comm_vec;

        // std::cout << std::endl;
        // for(u64 i = 0 ; i < comm_vec.size(); i++){
        //     std::cout << "(" << comm_vec[i].x() << ","
        //                       << comm_vec[i].y() << ") ";
        // }
        // std::cout << std::endl;

        // std::vector<u64_2> global_comm_vec;
        // mpi_handler::vector_allgatherv(comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2,
        // MPI_COMM_WORLD);

        // std::cout << std::endl;
        // for(u64 i = 0 ; i < global_comm_vec.size(); i++){
        //     std::cout << "(" << global_comm_vec[i].x() << ","
        //                       << global_comm_vec[i].y() << ") ";
        // }
        // std::cout << std::endl;
    }

    [[deprecated("Old sph module")]]
    inline static void comm_interface(
        PatchScheduler &sched, std::vector<InterfaceComm<vectype>> &interface_comm_list) {

        using namespace shamrock::patch;

        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataLayer>>>>
            Interface_map;

        for (const Patch &p : sched.patch_list.global) {
            Interface_map[p.id_patch]
                = std::vector<std::tuple<u64, std::unique_ptr<PatchDataLayer>>>();
        }

        std::vector<std::unique_ptr<PatchDataLayer>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count
                    > 0) {
                    std::vector<std::unique_ptr<PatchDataLayer>> pret
                        = InterfaceVolumeGenerator::append_interface<vectype>(
                            shamsys::instance::get_alt_queue(),
                            sched.patch_data.owned_data[interface_comm_list[i].sender_patch_id],
                            {interface_comm_list[i].interf_box_min},
                            {interface_comm_list[i].interf_box_max});
                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PatchDataLayer>());
                }
                comm_vec.push_back(
                    u64_2{
                        interface_comm_list[i].global_patch_idx_send,
                        interface_comm_list[i].global_patch_idx_recv});
            }

            std::cout << "\n split \n";
        }

        std::cout << "len comm_pdat : " << comm_pdat.size() << std::endl;
        std::cout << "len comm_vec : " << comm_vec.size() << std::endl;

        std::vector<i32> local_comm_tag(comm_vec.size());
        {
            i32 iterator = 0;
            for (u64 i = 0; i < comm_vec.size(); i++) {
                // const Patch & psend = sched.patch_list.global[comm_vec[i].x()];
                // const Patch & precv = sched.patch_list.global[comm_vec[i].y()];

                local_comm_tag[i] = iterator;

                iterator++;
            }
        }

        std::vector<u64_2> global_comm_vec;
        std::vector<i32> global_comm_tag;
        shamalgs::collective::vector_allgatherv(
            comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
        shamalgs::collective::vector_allgatherv(
            local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);

        std::vector<PatchDataMpiRequest> rq_lst;

        {
            for (u64 i = 0; i < comm_vec.size(); i++) {
                const Patch &psend = sched.patch_list.global[comm_vec[i].x()];
                const Patch &precv = sched.patch_list.global[comm_vec[i].y()];

                if (psend.node_owner_id == precv.node_owner_id) {
                    // std::cout << "same node !!!\n";
                    Interface_map[precv.id_patch].push_back(
                        {psend.id_patch, std::move(comm_pdat[i])});
                    comm_pdat[i] = nullptr;
                } else {
                    std::cout << shambase::format_printf(
                        "send : (%3d,%3d) : %d -> %d / %d\n",
                        psend.id_patch,
                        precv.id_patch,
                        psend.node_owner_id,
                        precv.node_owner_id,
                        local_comm_tag[i]);
                    patchdata_isend(
                        *comm_pdat[i],
                        rq_lst,
                        precv.node_owner_id,
                        local_comm_tag[i],
                        MPI_COMM_WORLD);
                }

                // std::cout << format("send : (%3d,%3d) : %d -> %d /
                // %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
                // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i],
                // MPI_COMM_WORLD);
            }
        }

        if (global_comm_vec.size() > 0) {

            // std::cout << std::endl;
            for (u64 i = 0; i < global_comm_vec.size(); i++) {

                const Patch &psend = sched.patch_list.global[global_comm_vec[i].x()];
                const Patch &precv = sched.patch_list.global[global_comm_vec[i].y()];
                // std::cout << format("(%3d,%3d) : %d -> %d /
                // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

                if (precv.node_owner_id == shamcomm::world_rank()) {

                    if (psend.node_owner_id != precv.node_owner_id) {
                        std::cout << shambase::format_printf(
                            "recv (%3d,%3d) : %d -> %d / %d\n",
                            global_comm_vec[i].x(),
                            global_comm_vec[i].y(),
                            psend.node_owner_id,
                            precv.node_owner_id,
                            global_comm_tag[i]);
                        Interface_map[precv.id_patch].push_back(
                            {psend.id_patch,
                             std::make_unique<PatchDataLayer>()}); // patchdata_irecv(recv_rq,
                                                                   // psend.node_owner_id,
                                                                   // global_comm_tag[i],
                                                                   // MPI_COMM_WORLD)}
                        patchdata_irecv_probe(
                            *std::get<1>(Interface_map[precv.id_patch]
                                                      [Interface_map[precv.id_patch].size() - 1]),
                            rq_lst,
                            psend.node_owner_id,
                            global_comm_tag[i],
                            MPI_COMM_WORLD);
                    }

                    // std::cout << format("recv (%3d,%3d) : %d -> %d /
                    // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                    // Interface_map[precv.id_patch].push_back({psend.id_patch, new
                    // PatchData()});//patchdata_irecv(recv_rq, psend.node_owner_id,
                    // global_comm_tag[i], MPI_COMM_WORLD)}
                    // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst,
                    // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }
            }
            // std::cout << std::endl;
        }

        waitall_pdat_mpi_rq(rq_lst);
    }
};

/*



template <class T> class OctreeMaxReducer {
public:
static T reduce(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {

    T tmp0 = sycl::max(v0, v1);
    T tmp1 = sycl::max(v2, v3);
    T tmp2 = sycl::max(v4, v5);
    T tmp3 = sycl::max(v6, v7);

    T tmpp0 = sycl::max(tmp0, tmp1);
    T tmpp1 = sycl::max(tmp2, tmp3);

    return sycl::max(tmpp0, tmpp1);
}
};
    inline void gen_interfaces(SchedulerMPI &sched, SerialPatchTree<vectype> &sptree,
                               PatchField<typename vectype::element_type> pfield) {



        const u64 local_pcount = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        sycl::buffer<u64> patch_ids_buf(local_pcount);
        sycl::buffer<vectype> local_box_min_buf(local_pcount);
        sycl::buffer<vectype> local_box_max_buf(local_pcount);
        sycl::buffer<u64_2, 2> interface_list_buf({local_pcount,global_pcount});
        sycl::buffer<u64, 2> stack_buf({local_pcount,global_pcount});
        sycl::buffer<u64> stack_start_idx_buf(local_pcount);

        {
            auto pid      = patch_ids_buf.get_access<sycl::access::mode::discard_write>();
            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read_write>();
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read_write>();

            auto stack = stack_buf.get_access<sycl::access::mode::discard_write>();
            auto stack_start_idx =
stack_start_idx_buf.get_access<sycl::access::mode::discard_write>();

            std::tuple<vectype,vectype> box_transform = sched.get_box_tranform<vectype>();

            for(u64 i = 0 ; i < local_pcount ; i ++){
                pid[i] = sched.patch_list.local[i].id_patch;

                lbox_min[i] = vectype{sched.patch_list.local[i].x_min,
sched.patch_list.local[i].y_min, sched.patch_list.local[i].z_min} * std::get<1>(box_transform) +
std::get<0>(box_transform); lbox_max[i] = (vectype{sched.patch_list.local[i].x_max,
sched.patch_list.local[i].y_max, sched.patch_list.local[i].z_max} + 1) * std::get<1>(box_transform)
+ std::get<0>(box_transform);

                lbox_min[i] -= pfield.local_nodes_value[i];
                lbox_max[i] += pfield.local_nodes_value[i];

                //TODO use root_ids list to init the stack instead of just 0
                stack[{i,0}] = 0;
                stack_start_idx[i] = 0;
            }
        }



        PatchFieldReduction<typename vectype::element_type> pfield_reduced =
            sptree.template reduce_field<typename vectype::element_type,
OctreeMaxReducer>(hndl.alt_queues[0], sched, pfield);



        hndl.alt_queues[0].submit([&](sycl::handler &cgh) {

            auto pid      = patch_ids_buf.get_access<sycl::access::mode::read>();
            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read>();
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read>();


            auto interface_list =
interface_list_buf.get_access<sycl::access::mode::discard_write>(cgh);

            auto stack = stack_buf.get_access<sycl::access::mode::read_write>();
            auto stack_start_idx = stack_start_idx_buf.get_access<sycl::access::mode::read>();


            cgh.parallel_for(sycl::range<1>(local_pcount), [=](sycl::item<1> item) {

                u64 cur_patch_idx = (u64)item.get_id(0);
                u64 cur_patch_id = pid[cur_patch_idx];
                u64 cur_lbox_min = lbox_min[cur_patch_idx];
                u64 cur_lbox_max = lbox_max[cur_patch_idx];


                u64 interface_ptr = 0;


                u64 current_stack_ptr = stack_start_idx[cur_patch_idx];

                while(current_stack_ptr != u64_max){

                    //pop stack
                    u64 cur_stack_idx = stack[{cur_patch_idx,current_stack_ptr}];
                    if(current_stack_ptr == 0){
                        current_stack_ptr = u64_max;
                    } else {
                        current_stack_ptr --;
                    }




                }

            });

        });

    }

*/

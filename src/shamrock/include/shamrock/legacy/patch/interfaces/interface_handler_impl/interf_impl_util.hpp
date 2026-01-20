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
 * @file interf_impl_util.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "interface_handler_impl_list.hpp"
#include "shamrock/legacy/utils/interact_crit_utils.hpp"
#include "shamtree/RadixTree.hpp"

namespace impl {

    template<class flt>
    struct CommInd {
        using vec = sycl::vec<flt, 3>;

        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        vec receiver_box_min;
        vec receiver_box_max;
    };

    namespace pfield_convertion {

        template<class T>
        struct accessed_pfield {
            sycl::accessor<T, 1, sycl::access::mode::read, sycl::target::device> acc_loc;
            sycl::accessor<T, 1, sycl::access::mode::read, sycl::target::device> acc_glo;

            inline accessed_pfield(BufferedPField<T> &pf, sycl::handler &cgh)
                : acc_loc(sycl::accessor{pf.buf_local, cgh, sycl::read_only}),
                  acc_glo(sycl::accessor{pf.buf_global, cgh, sycl::read_only}) {}

            inline T get_local(u32 i) const { return acc_loc[i]; }

            inline T get_global(u32 i) const { return acc_glo[i]; }
        };

    } // namespace pfield_convertion

    namespace generator {

        template<class flt>
        struct GeneratorBuffer {

            using vec = sycl::vec<flt, 3>;

            const u64 local_pcount;
            const u64 global_pcount;

            sycl::buffer<u64> patch_ids_buf;
            sycl::buffer<u64> global_ids_buf;
            sycl::buffer<vec> local_box_min_buf;
            sycl::buffer<vec> local_box_max_buf;

            sycl::buffer<vec> global_box_min_buf;
            sycl::buffer<vec> global_box_max_buf;

            explicit GeneratorBuffer(PatchScheduler &sched)
                : local_pcount(sched.patch_list.local.size()),
                  global_pcount(sched.patch_list.global.size()), patch_ids_buf(local_pcount),
                  global_ids_buf(global_pcount), local_box_min_buf(local_pcount),
                  local_box_max_buf(local_pcount), global_box_min_buf(global_pcount),
                  global_box_max_buf(global_pcount) {

                if (local_pcount == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "local patch count is zero this function can not run");
                }

                sycl::host_accessor pid{patch_ids_buf, sycl::write_only, sycl::no_init};
                sycl::host_accessor lbox_min{local_box_min_buf, sycl::write_only, sycl::no_init};
                sycl::host_accessor lbox_max{local_box_max_buf, sycl::write_only, sycl::no_init};

                sycl::host_accessor gbox_min{global_box_min_buf, sycl::write_only, sycl::no_init};
                sycl::host_accessor gbox_max{global_box_max_buf, sycl::write_only, sycl::no_init};

                std::tuple<vec, vec> box_transform = sched.get_box_tranform<vec>();

                for (u64 i = 0; i < local_pcount; i++) {
                    pid[i] = sched.patch_list.local[i].id_patch;

                    lbox_min[i] = vec{sched.patch_list.local[i].coord_min[0],
                                      sched.patch_list.local[i].coord_min[1],
                                      sched.patch_list.local[i].coord_min[2]}
                                      * std::get<1>(box_transform)
                                  + std::get<0>(box_transform);
                    lbox_max[i] = (vec{sched.patch_list.local[i].coord_max[0],
                                       sched.patch_list.local[i].coord_max[1],
                                       sched.patch_list.local[i].coord_max[2]}
                                   + 1)
                                      * std::get<1>(box_transform)
                                  + std::get<0>(box_transform);
                }

                // auto g_pid = global_ids_buf.get_access<sycl::access::mode::discard_write>();
                sycl::host_accessor g_pid{global_ids_buf, sycl::write_only, sycl::no_init};
                for (u64 i = 0; i < global_pcount; i++) {
                    g_pid[i] = sched.patch_list.global[i].id_patch;

                    gbox_min[i] = vec{sched.patch_list.global[i].coord_min[0],
                                      sched.patch_list.global[i].coord_min[1],
                                      sched.patch_list.global[i].coord_min[2]}
                                      * std::get<1>(box_transform)
                                  + std::get<0>(box_transform);
                    gbox_max[i] = (vec{sched.patch_list.global[i].coord_max[0],
                                       sched.patch_list.global[i].coord_max[1],
                                       sched.patch_list.global[i].coord_max[2]}
                                   + 1)
                                      * std::get<1>(box_transform)
                                  + std::get<0>(box_transform);
                }
            };
        };

        template<class flt, class InteractCd, class... Args>
        inline sycl::buffer<impl::CommInd<flt>, 2> compute_buf_interact(
            PatchScheduler &sched,
            GeneratorBuffer<flt> &gen,
            SerialPatchTree<sycl::vec<flt, 3>> &sptree,
            sycl::vec<flt, 3> test_patch_offset,
            bool has_off,
            const InteractCd &interact_crit,
            Args... args) {

            using vec = sycl::vec<flt, 3>;

            const u64 &local_pcount  = gen.local_pcount;
            const u64 &global_pcount = gen.global_pcount;

            shamlog_debug_sycl_ln(
                "InterfaceFinder",
                "searching interfaces offset=",
                test_patch_offset,
                "loc count =",
                local_pcount,
                "g count=",
                global_pcount);

            sycl::buffer<u64> &patch_ids_buf      = gen.patch_ids_buf;
            sycl::buffer<u64> &global_ids_buf     = gen.global_ids_buf;
            sycl::buffer<vec> &local_box_min_buf  = gen.local_box_min_buf;
            sycl::buffer<vec> &local_box_max_buf  = gen.local_box_max_buf;
            sycl::buffer<vec> &global_box_min_buf = gen.global_box_min_buf;
            sycl::buffer<vec> &global_box_max_buf = gen.global_box_max_buf;

            sycl::buffer<impl::CommInd<flt>, 2> interface_list_buf({local_pcount, global_pcount});

            // was used for smoothing length
            // sycl::buffer<flt>
            // buf_local_field_val(pfield.local_nodes_value.data(),pfield.local_nodes_value.size());
            // sycl::buffer<flt>
            // buf_global_field_val(pfield.global_values.data(),pfield.global_values.size());

            shamsys::instance::get_alt_queue().submit([&](sycl::handler &cgh) {
                auto compute_interf = [&](auto... acc_fields) {
                    auto pid  = sycl::accessor{patch_ids_buf, cgh, sycl::read_only};
                    auto gpid = sycl::accessor{global_ids_buf, cgh, sycl::read_only};

                    auto lbox_min = sycl::accessor{local_box_min_buf, cgh, sycl::read_only};
                    auto lbox_max = sycl::accessor{local_box_max_buf, cgh, sycl::read_only};

                    auto gbox_min = sycl::accessor{global_box_min_buf, cgh, sycl::read_only};
                    auto gbox_max = sycl::accessor{global_box_max_buf, cgh, sycl::read_only};

                    auto interface_list
                        = interface_list_buf.template get_access<sycl::access::mode::discard_write>(
                            cgh);

                    u64 cnt_patch = global_pcount;

                    vec offset = test_patch_offset;

                    bool is_off_not_bull = has_off;

                    InteractCd cd = interact_crit;

                    // auto out = sycl::stream(4096*4, 4096, cgh);

                    cgh.parallel_for(sycl::range<1>(local_pcount), [=](sycl::item<1> item) {
                        u64 cur_patch_idx = (u64) item.get_id(0);
                        u64 cur_patch_id  = pid[cur_patch_idx];
                        vec cur_lbox_min  = lbox_min[cur_patch_idx];
                        vec cur_lbox_max  = lbox_max[cur_patch_idx];

                        // out << "-> "<<cur_patch_id<<" : "<<cur_lbox_min << " | " << cur_lbox_max
                        // <<"\n";

                        u64 interface_ptr = 0;

                        for (u64 test_patch_idx = 0; test_patch_idx < cnt_patch; test_patch_idx++) {

                            // keep in mind that we compute patch that we have to send
                            // so we apply this offset on the patch we test against rather than ours
                            vec test_lbox_min = gbox_min[test_patch_idx] + offset;
                            vec test_lbox_max = gbox_max[test_patch_idx] + offset;
                            u64 test_patch_id = gpid[test_patch_idx];

                            {

                                bool is_not_itself
                                    = (is_off_not_bull) || (test_patch_id != cur_patch_id);

#if false
                                out << "   testing : "<<test_patch_id << "\n";
                                auto interact_cd_cell_patch = [out](const auto & cd,vec b1_min, vec b1_max,vec b2_min, vec b2_max, flt b1_min_slength, flt b1_max_slength, flt b2_min_slength, flt b2_max_slength){


                                    out << "\t\tb1_min : "<<b1_min << "\n";
                                    out << "\t\tb1_max : "<<b1_max << "\n";
                                    out << "\t\tb2_min : "<<b2_min << "\n";
                                    out << "\t\tb2_max : "<<b2_max << "\n";


                                    out << "\t\tb2_min_slength : "<<b2_min_slength << "\n";
                                    out << "\t\tb2_max_slength : "<<b2_max_slength << "\n";


                                    vec c1 = (b1_max + b1_min)/2;
                                    vec s1 = (b1_max - b1_min);
                                    flt L1 = sycl::max(sycl::max(s1.x(),s1.y()),s1.z());

                                    flt dist_to_surf = sycl::sqrt(BBAA::get_sq_distance_to_BBAAsurface(c1, b2_min, b2_max));


                                    out << "\t\tdist_to_surf : "<<dist_to_surf << "\n";
                                    out << "\t\ttop : "<<L1 + b2_max_slength << "\n";
                                    out << "\t\tbot : "<<dist_to_surf + b2_min_slength/2 << "\n";

                                    flt opening_angle_sq = (L1 + b2_max_slength)/(dist_to_surf + b2_min_slength/2);
                                    opening_angle_sq *= opening_angle_sq;

                                    out << "\t\tangle : "<<sycl::sqrt(opening_angle_sq) << "\n";

                                    return opening_angle_sq > 0.5*0.5;
                                };
#endif

                                // check if us (cur_patch_id) : (patch) interact with any of the
                                // leafs of the (other) traget patch (eg test_patch_id) so the
                                // relation is : R(Sender, U receiver leaf)  aka
                                // interact_cd_cell_patch
                                // TODO interact_cd_cell_patch is confusing : cell <=> root cell of
                                // the patch => unclear
                                bool int_crit = interact_crit::utils::interact_cd_cell_patch_domain(
                                    cd,
                                    is_off_not_bull,
                                    cur_lbox_min,
                                    cur_lbox_max,
                                    test_lbox_min,
                                    test_lbox_max,
                                    acc_fields.get_local(cur_patch_idx)...,
                                    acc_fields.get_global(test_patch_idx)...);

#if false
                                interact_cd_cell_patch(
                                    cd,
                                    cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                    acc_fields.get_local(cur_patch_idx)...,acc_fields.get_global(test_patch_idx)...
                                    );
#endif

                                if (int_crit && is_not_itself) {

                                    // out << "detected : -> "<<test_patch_id<<"\n";

                                    interface_list[{cur_patch_idx, interface_ptr}]
                                        = impl::CommInd<flt>{
                                            cur_patch_idx,
                                            test_patch_idx,
                                            cur_patch_id,
                                            test_patch_id,
                                            test_lbox_min,
                                            test_lbox_max};

                                    interface_ptr++;
                                }
                            }
                        }

                        if (interface_ptr < global_pcount) {
                            interface_list[{cur_patch_idx, interface_ptr}] = impl::CommInd<flt>{
                                u64_max, u64_max, u64_max, u64_max, vec{0, 0, 0}, vec{0, 0, 0}};
                        }
                    });
                };

                compute_interf(impl::pfield_convertion::accessed_pfield{args.patch_field, cgh}...);
            });

            //{
            //    sycl::host_accessor acc{interface_list_buf};
            //    for(u32 i = 0; i < local_pcount; i++){
            //        for(u32 j = 0; j < global_pcount; j++){
            //            std::cout << acc[{i,j}].global_patch_idx_recv << " ";
            //        }std::cout << "\n";
            //    }
            //}std::cout << std::endl;

            return std::move(interface_list_buf);
        }

    } // namespace generator

} // namespace impl

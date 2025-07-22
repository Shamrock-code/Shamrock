// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NeighGraphLinkField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class NeighGraphLinkField {
        public:
        sham::DeviceBuffer<T> link_graph_field;
        u32 link_count;
        u32 nvar;

        void resize(NeighGraph &graph) {
            if (link_count != graph.link_count) {
                link_count = graph.link_count;
                link_graph_field.resize(link_count * nvar);
            }
        }
        void resize(u32 count) {
            if (link_count != count) {
                link_count = count;
                link_graph_field.resize(link_count * nvar);
            }
        }

        NeighGraphLinkField(u32 nvar)
            : link_graph_field(0, shamsys::instance::get_compute_scheduler_ptr()), nvar(nvar),
              link_count(0) {}

        NeighGraphLinkField(NeighGraph &graph)
            : link_graph_field(graph.link_count, shamsys::instance::get_compute_scheduler_ptr()),
              link_count(graph.link_count), nvar(1) {}

        NeighGraphLinkField(NeighGraph &graph, u32 nvar)
            : link_graph_field(
                  graph.link_count * nvar, shamsys::instance::get_compute_scheduler_ptr()),
              link_count(graph.link_count), nvar(nvar) {}

        NeighGraphLinkField(u32 link_count, u32 nvar)
            : link_graph_field(link_count * nvar, shamsys::instance::get_compute_scheduler_ptr()),
              link_count(link_count), nvar(nvar) {}

        inline auto get_read_access(sham::EventList &deps) const {
            return link_graph_field.get_read_access(deps);
        }
        inline auto get_write_access(sham::EventList &deps) {
            return link_graph_field.get_write_access(deps);
        }
        inline void complete_event_state(sycl::event e) const {
            return link_graph_field.complete_event_state(e);
        }
    };

    template<class LinkFieldCompute, class T>
    inline void ddupdate_link_field(
        sham::DeviceScheduler_ptr dev_sched,
        shambase::DistributedData<NeighGraphLinkField<T>> &neigh_graph_field,
        shambase::DistributedData<NeighGraph> &graph,
        shambase::DistributedData<LinkFieldCompute> &fcomp) {
        StackEntry stack_loc{};

        auto &result = neigh_graph_field;

        shambase::DistributedData<u32> counts = graph.map<u32>([&](u64 id, u32 block_count) {
            return graph.get(id).obj_cnt;
        });

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{graph, fcomp},
            sham::DDMultiRef{neigh_graph_field},
            counts,
            [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
                link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
                });
            });
    }

    template<class LinkFieldCompute, class T, class... Args>
    inline void update_link_field(
        sham::DeviceQueue &q,
        sham::EventList &depends_list,
        sham::EventList &result_list,
        NeighGraphLinkField<T> &neigh_graph_field,
        NeighGraph &graph,
        Args &&...args) {
        StackEntry stack_loc{};

        auto &result = neigh_graph_field;

        result.resize(graph);

        auto acc_link_field = result.link_graph_field.get_write_access(depends_list);
        auto link_iter      = graph.get_read_access(depends_list);

        LinkFieldCompute compute(std::forward<Args>(args)...);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parralel_for(cgh, graph.obj_cnt, "compute link field", [=](u32 id_a) {
                link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
                });
            });
        });

        result_list.add_event(e);
        result.link_graph_field.complete_event_state(e);
        graph.complete_event_state(e);
    }
    template<class LinkFieldCompute, class T, class... Args>
    inline void update_link_field_indep_nvar(
        sham::DeviceQueue &q,
        sham::EventList &depends_list,
        sham::EventList &result_list,
        NeighGraphLinkField<T> &neigh_graph_field,
        NeighGraph &graph,
        u32 nvar,
        Args &&...args) {
        StackEntry stack_loc{};

        auto &result = neigh_graph_field;

        result.resize(graph);

        auto acc_link_field = result.link_graph_field.get_write_access(depends_list);
        auto link_iter      = graph.get_read_access(depends_list);

        LinkFieldCompute compute(nvar, std::forward<Args>(args)...);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parralel_for(
                cgh, graph.obj_cnt * nvar, "compute link field indep nvar", [=](u32 idvar_a) {
                    const u32 id_cell_a = idvar_a / nvar;
                    const u32 nvar_loc  = idvar_a % nvar;

                    link_iter.for_each_object_link_id(id_cell_a, [&](u32 id_cell_b, u32 link_id) {
                        acc_link_field[link_id * nvar + nvar_loc] = compute.get_link_field_val(
                            id_cell_a * nvar + nvar_loc, id_cell_b * nvar + nvar_loc);
                    });
                });
        });

        result_list.add_event(e);
        result.link_graph_field.complete_event_state(e);
        graph.complete_event_state(e);
    }

    template<class LinkFieldCompute, class T, class... Args>
    NeighGraphLinkField<T> compute_link_field(
        sham::DeviceQueue &q,
        sham::EventList &depends_list,
        sham::EventList &result_list,
        NeighGraph &graph,
        Args &&...args) {
        StackEntry stack_loc{};

        NeighGraphLinkField<T> result{graph};

        auto acc_link_field = result.link_graph_field.get_write_access(depends_list);
        auto link_iter      = graph.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            LinkFieldCompute compute(cgh, std::forward<Args>(args)...);

            shambase::parralel_for(cgh, graph.obj_cnt, "compute link field", [=](u32 id_a) {
                link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
                });
            });
        });

        result_list.add_event(e);
        result.link_graph_field.complete_event_state(e);
        graph.complete_event_state(e);

        return result;
    }

    template<class LinkFieldCompute, class T, class... Args>
    NeighGraphLinkField<T> compute_link_field_indep_nvar(
        sham::DeviceQueue &q,
        sham::EventList &depends_list,
        sham::EventList &result_list,
        NeighGraph &graph,
        u32 nvar,
        Args &&...args) {

        StackEntry stack_loc{};

        NeighGraphLinkField<T> result{graph, nvar};

        auto acc_link_field = result.link_graph_field.get_write_access(depends_list);
        auto link_iter      = graph.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            LinkFieldCompute compute(cgh, nvar, std::forward<Args>(args)...);

            shambase::parralel_for(
                cgh, graph.obj_cnt * nvar, "compute link field indep nvar", [=](u32 idvar_a) {
                    const u32 id_cell_a = idvar_a / nvar;
                    const u32 nvar_loc  = idvar_a % nvar;

                    link_iter.for_each_object_link_id(id_cell_a, [&](u32 id_cell_b, u32 link_id) {
                        acc_link_field[link_id * nvar + nvar_loc] = compute.get_link_field_val(
                            id_cell_a * nvar + nvar_loc, id_cell_b * nvar + nvar_loc);
                    });
                });
        });

        result_list.add_event(e);
        result.link_graph_field.complete_event_state(e);
        graph.complete_event_state(e);

        return result;
    }

    /*
    template<class Tvec>
    class FaceShiftInfo{
        sycl::buffer<Tvec> link_shift_a;
        sycl::buffer<Tvec> link_shift_b;
        u32 link_count;

        FaceShiftInfo(NeighGraph & graph):
        link_shift_a(graph.link_count),
        link_shift_b(graph.link_count),
        link_count(graph.link_count) {}
    };

    template<class Tvec, class TgridVec, class AMRBLock>
    FaceShiftInfo<Tvec> get_face_shift_infos(sycl::queue &q,NeighGraph & graph,
            sycl::buffer<TgridVec> &buf_block_min,
            sycl::buffer<TgridVec> &buf_block_max,
            ){

        FaceShiftInfo<Tvec> shifts (graph);

        q.submit([&](sycl::handler &cgh) {
            NeighGraphLinkiterator link_iter {graph, cgh};
            LinkFieldCompute compute (cgh, std::forward<Args>(args)...);

            sycl::accessor acc_link_field {result.template link_graph_field, cgh, sycl::write_only,
    sycl::no_init};

            shambase::parralel_for(cgh, graph.obj_cnt, "compute link field", [=](u32 id_a) {

                link_iter.for_each_object_link(id_a, [&](u32 id_b, u32 link_id){
                    acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
                });

            });
        });

        return shifts;

    }
    */

} // namespace shammodels::basegodunov::modules

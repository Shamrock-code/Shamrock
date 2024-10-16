// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NeighGraphLinkField.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include "shammodels/amr/NeighGraph.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class NeighGraphLinkField {
        public:
        sycl::buffer<T> link_graph_field;
        u32 link_count;
        u32 nvar;

        NeighGraphLinkField(NeighGraph &graph)
            : link_graph_field(graph.link_count), link_count(graph.link_count), nvar(1) {}

        NeighGraphLinkField(NeighGraph &graph, u32 nvar)
            : link_graph_field(graph.link_count * nvar), link_count(graph.link_count), nvar(nvar) {}
    };

    template<class LinkFieldCompute, class T, class... Args>
    NeighGraphLinkField<T> compute_link_field(sycl::queue &q, NeighGraph &graph, Args &&...args) {

        NeighGraphLinkField<T> result{graph};

        q.submit([&](sycl::handler &cgh) {
            NeighGraphLinkiterator link_iter{graph, cgh};
            LinkFieldCompute compute(cgh, std::forward<Args>(args)...);

            sycl::accessor acc_link_field{
                result.link_graph_field, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, graph.obj_cnt, "compute link field", [=](u32 id_a) {
                link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                    acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
                });
            });
        });

        return result;
    }
    template<class LinkFieldCompute, class T, class... Args>
    NeighGraphLinkField<T>
    compute_link_field_indep_nvar(sycl::queue &q, NeighGraph &graph, u32 nvar, Args &&...args) {

        NeighGraphLinkField<T> result{graph, nvar};

        q.submit([&](sycl::handler &cgh) {
            NeighGraphLinkiterator link_iter{graph, cgh};
            LinkFieldCompute compute(cgh, nvar, std::forward<Args>(args)...);

            sycl::accessor acc_link_field{
                result.link_graph_field, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(
                cgh, graph.obj_cnt * nvar, "compute link field indep nvar", [=](u32 idvar_a) {
                    const u32 id_cell_a = idvar_a / nvar;
                    const u32 nvar_loc  = idvar_a % nvar;

                    link_iter.for_each_object_link_id(id_cell_a, [&](u32 id_cell_b, u32 link_id) {
                        acc_link_field[link_id * nvar + nvar_loc] = compute.get_link_field_val(
                            id_cell_a * nvar + nvar_loc,
                            id_cell_b * nvar + nvar_loc); // id_cell_a ==> link_id
                    });
                });
        });

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

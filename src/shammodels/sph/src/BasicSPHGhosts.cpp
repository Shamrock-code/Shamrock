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

/*

Test code for godbolt


#include <iostream>
#include <vector>

namespace sycl{
    template<class T>
    struct vec{
        T _x,_y,_z;

        inline T & x(){
            return _x;
        }

        inline T & y(){
            return _y;
        }
        inline T & z(){
            return _z;
        }
    };
}


using i32 = int;
using i32_3 = sycl::vec<i32>;

template<class T>
struct ShiftInfo{
    sycl::vec<T> shift;
    sycl::vec<T> shift_speed;
};

template<class T>
struct ShearPeriodicInfo{
    i32_3 shear_base;
    i32_3 shear_dir;
    T shear_value;
    T shear_speed;
};

template<class T>
inline ShiftInfo<T> compute_shift_infos(
    i32_3 ioff, ShearPeriodicInfo<T> shear, sycl::vec<T> bsize
    ){

    i32 dx = ioff.x()*shear.shear_base.x();
    i32 dy = ioff.y()*shear.shear_base.y();
    i32 dz = ioff.z()*shear.shear_base.z();

    i32 d = dx + dy + dz;

    sycl::vec<T> shift = {
        (d*shear.shear_dir.x())*shear.shear_value + bsize.x()*ioff.x(),
        (d*shear.shear_dir.y())*shear.shear_value + bsize.y()*ioff.y() ,
        (d*shear.shear_dir.z())*shear.shear_value + bsize.z()*ioff.z()
    };
    sycl::vec<T> shift_speed = {
        (d*shear.shear_dir.x())*shear.shear_speed,
        (d*shear.shear_dir.y())*shear.shear_speed,
        (d*shear.shear_dir.z())*shear.shear_speed
    };

    return {shift,shift_speed};
}

template<class T>
inline void for_each_patch_shift(ShearPeriodicInfo<T> shearinfo, sycl::vec<T> bsize){

    i32_3 loop_offset = {0,0,0};

    std::vector<i32_3> list_possible;


    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;



    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {


                i32 dx = xoff*shearinfo.shear_base.x();
                i32 dy = yoff*shearinfo.shear_base.y();
                i32 dz = zoff*shearinfo.shear_base.z();

                i32 d = dx + dy + dz;

                i32 df = -int(d * shearinfo.shear_value);

                i32_3 off_d = {
                    shearinfo.shear_dir.x()*df,
                    shearinfo.shear_dir.y()*df,
                    shearinfo.shear_dir.z()*df
                };

                list_possible.push_back({xoff+off_d.x(),yoff+off_d.y(),zoff+off_d.z()});
            }
        }
    }

    for(i32_3 off : list_possible){

        auto shift = compute_shift_infos(off,shearinfo,bsize);

        std::cout <<
            off.x() << " " << off.y() << " " << off.z() << " | " <<
            shift.shift.x() << " " << shift.shift.y() << " " << shift.shift.z() << " "<<std::endl;
    }



}


int main(){

    ShearPeriodicInfo<float> shear{
        {1,0,0},
        {0,0,1},
        13.5,
        1
    };

    for_each_patch_shift(shear, {1,1,1});

}


*/

#include "shambase/exception.hpp"
#include "shamcomm/collectives.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"
#include <functional>
#include <vector>

template<class T>
struct ShiftInfo {
    sycl::vec<T, 3> shift;
    sycl::vec<T, 3> shift_speed;
};

template<class T>
using ShearPeriodicInfo =
    typename shammodels::sph::BasicSPHGhostHandlerConfig<sycl::vec<T, 3>>::ShearingPeriodic;

template<class T>
inline ShiftInfo<T> compute_shift_infos(
    i32_3 ioff, ShearPeriodicInfo<T> shear, sycl::vec<T, 3> bsize) {

    i32 dx = ioff.x() * shear.shear_base.x();
    i32 dy = ioff.y() * shear.shear_base.y();
    i32 dz = ioff.z() * shear.shear_base.z();

    i32 d = dx + dy + dz;

    sycl::vec<T, 3> shift
        = {(d * shear.shear_dir.x()) * shear.shear_value + bsize.x() * ioff.x(),
           (d * shear.shear_dir.y()) * shear.shear_value + bsize.y() * ioff.y(),
           (d * shear.shear_dir.z()) * shear.shear_value + bsize.z() * ioff.z()};
    sycl::vec<T, 3> shift_speed
        = {(d * shear.shear_dir.x()) * shear.shear_speed,
           (d * shear.shear_dir.y()) * shear.shear_speed,
           (d * shear.shear_dir.z()) * shear.shear_speed};

    return {shift, shift_speed};
}

template<class T>
inline void for_each_patch_shift(
    ShearPeriodicInfo<T> shearinfo,
    sycl::vec<T, 3> bsize,
    std::function<void(i32_3, ShiftInfo<T>)> funct) {

    i32_3 loop_offset = {0, 0, 0};

    std::vector<i32_3> list_possible;

    // logger::raw_ln("testing :",shearinfo.shear_value,shearinfo.shear_dir, shearinfo.shear_base);

    // a bit of dirty fix doesn't hurt
    // this should be done in a better way a some point
    i32 repetition_x = 1 + abs(shearinfo.shear_dir.x());
    i32 repetition_y = 1 + abs(shearinfo.shear_dir.y());
    i32 repetition_z = 1 + abs(shearinfo.shear_dir.z());

    T sz = bsize.x() * shearinfo.shear_dir.x() + bsize.y() * shearinfo.shear_dir.y()
           + bsize.z() * shearinfo.shear_dir.z();

    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                i32 dx = xoff * shearinfo.shear_base.x();
                i32 dy = yoff * shearinfo.shear_base.y();
                i32 dz = zoff * shearinfo.shear_base.z();

                i32 d = dx + dy + dz;

                i32 df = -int(d * shearinfo.shear_value / sz);

                i32_3 off_d
                    = {shearinfo.shear_dir.x() * df,
                       shearinfo.shear_dir.y() * df,
                       shearinfo.shear_dir.z() * df};

                // on redhat based systems stl vector freaks out
                // because iterator to back does *(end() - 1)
                // the issue is that the compiler gets confused
                // by the sycl::vec defining the - operator
                // creating the ambiguity and ...
                // ultimatly the compiler shitting itself
                list_possible.resize(list_possible.size() + 1);
                list_possible[list_possible.size() - 1]
                    = i32_3{xoff + off_d.x(), yoff + off_d.y(), zoff + off_d.z()};
            }
        }
    }

    // logger::raw_ln("trying", list_possible.size(), "patches ghosts");

    for (i32_3 off : list_possible) {

        auto shift = compute_shift_infos(off, shearinfo, bsize);

        // logger::raw_ln("check :",off,shift.shift, shift.shift_speed);

        funct(off, shift);
    }
}

namespace shammodels::sph::modules {
    enum class GhostType { None, Periodic, Reflective, ShearingPeriodic };

    struct GhostLayerGenMode {
        GhostType ghost_type_x;
        GhostType ghost_type_y;
        GhostType ghost_type_z;
    };

    template<class Tvec, class Tscal>
    shammath::paving_function_general_3d<Tvec> get_paving_with_no_shear(
        GhostLayerGenMode mode, shammath::AABB<Tvec> sim_box) {

        Tvec box_size   = sim_box.upper - sim_box.lower;
        Tvec box_center = (sim_box.upper + sim_box.lower) / 2;

        SHAM_ASSERT(sim_box.is_volume_not_null());

        { // check that rebuildind the AABB from size and center gives the same AABB
            shammath::AABB<Tvec> new_box = {box_center - box_size / 2, box_center + box_size / 2};
            if (new_box != sim_box) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Rebuilding AABB from size and center gives a different AABB");
            }
        }

        return shammath::paving_function_general_3d<Tvec>{
            box_size,
            box_center,
            mode.ghost_type_x == GhostType::Periodic,
            mode.ghost_type_y == GhostType::Periodic,
            mode.ghost_type_z == GhostType::Periodic};
    }

    template<class Tvec, class Tscal>
    shammath::paving_function_general_3d_shear_x<Tvec> get_paving_with_shear(
        GhostLayerGenMode mode, shammath::AABB<Tvec> sim_box, Tscal shear_x) {

        static_assert(
            std::is_same_v<Tscal, shambase::VecComponent<Tvec>>,
            "Tscal must be a vector component");

        Tvec box_size   = sim_box.upper - sim_box.lower;
        Tvec box_center = (sim_box.upper + sim_box.lower) / 2;

        SHAM_ASSERT(sim_box.is_volume_not_null());

        { // check that rebuildind the AABB from size and center gives the same AABB
            shammath::AABB<Tvec> new_box = {box_center - box_size / 2, box_center + box_size / 2};
            if (new_box != sim_box) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Rebuilding AABB from size and center gives a different AABB");
            }
        }

        return shammath::paving_function_general_3d_shear_x<Tvec>{
            box_size,
            box_center,
            mode.ghost_type_x == GhostType::Periodic,
            mode.ghost_type_y == GhostType::Periodic,
            mode.ghost_type_z == GhostType::Periodic,
            shear_x};
    }

    template<class Func>
    void for_each_paving_tile(GhostLayerGenMode mode, Func &&func) {

        // if the ghost type is none, we do not need to repeat as there is no ghost layer
        i32 repetition_x = mode.ghost_type_x != GhostType::None;
        i32 repetition_y = mode.ghost_type_y != GhostType::None;
        i32 repetition_z = mode.ghost_type_z != GhostType::None;

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {
                    func(xoff, yoff, zoff);
                }
            }
        }
    }

    struct GhostLayerCandidateInfos {
        i32 xoff;
        i32 yoff;
        i32 zoff;
    };

    template<class Tvec>
    class FindGhostLayerCandidatesSPH : public shamrock::solvergraph::INode {

        public:
        using Tscal = shambase::VecComponent<Tvec>;

        FindGhostLayerCandidatesSPH(GhostLayerGenMode mode, std::optional<Tscal> shear_x)
            : mode(mode), shear_x(shear_x) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::IDataEdge<std::vector<u64>> &ids_to_check;
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>> &sim_box;
            const shamrock::solvergraph::SerialPatchTreeRefEdge<Tvec> &patch_tree;
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>> &patch_boxes;
            const shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal> &int_range_max;
            const shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>
                &int_range_max_serialized;
            // outputs
            shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>
                &ghost_layers_candidates;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<std::vector<u64>>> ids_to_check,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::SerialPatchTreeRefEdge<Tvec>> patch_tree,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>>> patch_boxes,
            std::shared_ptr<shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal>> int_range_max,
            std::shared_ptr<shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>>
                int_range_max_serialized,
            std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
                ghost_layers_candidates) {
            __internal_set_ro_edges(
                {ids_to_check,
                 sim_box,
                 patch_tree,
                 patch_boxes,
                 int_range_max,
                 int_range_max_serialized});
            __internal_set_rw_edges({ghost_layers_candidates});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<std::vector<u64>>>(0),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>>>(1),
                get_ro_edge<shamrock::solvergraph::SerialPatchTreeRefEdge<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>>>(3),
                get_ro_edge<shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>>(5),
                get_rw_edge<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "FindGhostLayerCandidatesSPH";
        };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };

        private:
        GhostLayerGenMode mode;
        std::optional<Tscal> shear_x;
    };

    template<class Tvec>
    void shammodels::sph::modules::FindGhostLayerCandidatesSPH<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        // inputs
        auto &ids_to_check       = edges.ids_to_check.data;
        auto &sim_box            = edges.sim_box.value;
        auto &patch_tree         = edges.patch_tree.get_patch_tree();
        auto &patch_boxes        = edges.patch_boxes;
        auto &int_range_max      = edges.int_range_max.get_patch_tree_field();
        auto &int_range_max_tree = edges.int_range_max_tree.get_patch_tree_field();

        using PtNode = typename SerialPatchTree<Tvec>::PtNode;

        // outputs
        auto &ghost_layers_candidates = edges.ghost_layers_candidates.values;

        auto run_gz_search = [&](auto &paving, auto &for_each_paving_tile) {
            using namespace shamrock::patch;

            // for each repetitions
            for_each_paving_tile(mode, [&](i32 xoff, i32 yoff, i32 zoff) {
                // for all local patches
                for (auto id : ids_to_check) {
                    auto patch_box = patch_boxes.values.get(id);

                    Tscal sender_h_max = int_range_max.get(id);

                    // f(patch)
                    auto patch_box_mapped = paving.f_aabb(patch_box, xoff, yoff, zoff);

                    patch_tree.host_for_each_leafs(
                        [&](u64 tree_id, PtNode n) {
                            // here we use the group group interaction criteria AABB1 V (AABB2 +
                            // max(h_max1, h_max2))

                            shammath::AABB<Tvec> tree_cell{n.box_min, n.box_max};
                            Tscal receiv_h_max = int_range_max_tree.get(tree_id);

                            tree_cell = tree_cell.expand_all(sham::max(sender_h_max, receiv_h_max));

                            // f(patch) V box =! empty (a surface is not an empty set btw)
                            // <=> is ghost layer != empty
                            return tree_cell.get_intersect(patch_box_mapped).is_not_empty();
                        },
                        [&](u64 id_found, PtNode n) {
                            // skip self intersection (but not if we are through a boundary)
                            if ((id_found == id) && (xoff == 0) && (yoff == 0) && (zoff == 0)) {
                                return;
                            }

                            // we have an ghost layer between
                            // patch `id` and patch `id_found` for this offset
                            // so we store that
                            ghost_layers_candidates.add_obj(
                                id, id_found, GhostLayerCandidateInfos{xoff, yoff, zoff});

                            // TODO: we should suppress this add if the source or dest is empty
                        });
                }
            });
        };

        if (shear_x.has_value()) {
            auto paving = get_paving_with_shear(mode, sim_box, shear_x.value());
            run_gz_search(
                paving,
                for_each_paving_tile); // TODO: we need to offset the x index during search when
                                       // shearing
        } else {
            auto paving = get_paving_with_no_shear(mode, sim_box);
            run_gz_search(paving, for_each_paving_tile);
        }
    }

    template<class Tvec>
    class FindGhostLayerIndices : public shamrock::solvergraph::INode {

        public:
        using Tscal = shambase::VecComponent<Tvec>;
        FindGhostLayerIndices(GhostLayerGenMode mode, u32 ixyz, u32 ihpart)
            : mode(mode), ixyz(ixyz), ihpart(ihpart) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>> &sim_box;
            const shamrock::solvergraph::IPatchDataLayerRefs &patch_data_layers;
            const shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>
                &ghost_layers_candidates;
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>> &patch_boxes;
            const shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal> &int_range_max;
            const shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>
                &int_range_max_serialized;
            // outputs
            shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> patch_data_layers,
            std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
                ghost_layers_candidates,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>>> patch_boxes,
            std::shared_ptr<shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal>> int_range_max,
            std::shared_ptr<shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>>
                int_range_max_serialized,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost) {
            __internal_set_ro_edges(
                {sim_box,
                 patch_data_layers,
                 ghost_layers_candidates,
                 patch_boxes,
                 int_range_max,
                 int_range_max_serialized});
            __internal_set_rw_edges({idx_in_ghost});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>>>(0),
                get_ro_edge<shamrock::solvergraph::IPatchDataLayerRefs>(1),
                get_ro_edge<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<Tvec>>>(3),
                get_ro_edge<shamrock::solvergraph::SerialPatchFieldRefEdge<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::SerialPatchTreeFieldRefEdge<Tscal>>(5),
                get_rw_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "FindGhostLayerIndices"; };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };

        private:
        GhostLayerGenMode mode;
        std::optional<Tscal> shear_x;
        u32 ixyz;
        u32 ihpart;
    };

    template<class Tvec, class Tscal>
    sham::DeviceBuffer<u32> get_ids_in_ghost(
        const sham::DeviceScheduler_ptr &dev_sched,
        sham::DeviceQueue &q,
        const sham::DeviceBuffer<Tvec> &xyz,
        const sham::DeviceBuffer<Tscal> &hpart,
        const shammath::AABB<Tvec> &test_volume,
        const shammath::AABB<Tscal> &volume_hmax,
        u32 obj_cnt) {

        if (obj_cnt > 0) {
            // buffer of booleans to store result of the condition
            sham::DeviceBuffer<u32> mask(obj_cnt, dev_sched);

            sham::kernel_call(
                q,
                sham::MultiRef{xyz, hpart},
                sham::MultiRef{mask},
                obj_cnt,
                [test_volume, volume_hmax](
                    u32 id,
                    const Tvec *__restrict xyz,
                    const Tscal *__restrict hpart,
                    u32 *__restrict is_in_ghost) {
                    Tvec xyz_1 = xyz[id];
                    Tscal h_1  = hpart[id];

                    // object group interaction criteria {xyz_1} V (AABB2 + max(h_1, h_max2))
                    is_in_ghost[id] = test_volume.expand_all(sham::max(h_1, volume_hmax))
                                          .contains_symmetric(xyz_1);
                });

            return shamalgs::stream_compact(dev_sched, mask, obj_cnt);
        } else {
            return sham::DeviceBuffer<u32>(0, dev_sched);
        }
    }

    template<class Tvec>
    void shammodels::sph::modules::FindGhostLayerIndices<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        // inputs
        auto &sim_box                 = edges.sim_box.value;
        auto &patch_data_layers       = edges.patch_data_layers;
        auto &ghost_layers_candidates = edges.ghost_layers_candidates;
        auto &patch_boxes             = edges.patch_boxes;
        auto &int_range_max           = edges.int_range_max.get_patch_tree_field();
        auto &int_range_max_tree      = edges.int_range_max_tree.get_patch_tree_field();

        // outputs
        auto &idx_in_ghost = edges.idx_in_ghost;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

        auto &patch_data_layers_ref = patch_data_layers.get_const_refs();

        auto run_id_find = [&](auto &paving_function) {
            // map candidates to indexes in ghosts
            idx_in_ghost.buffers
                = ghost_layers_candidates.values.template map<sham::DeviceBuffer<u32>>(
                    [&](u64 sender,
                        u64 receiver,
                        const GhostLayerCandidateInfos &infos) -> sham::DeviceBuffer<u32> {
                        shamrock::patch::PatchDataLayer &sender_patch
                            = patch_data_layers_ref.get(sender).get();

                        PatchDataField<Tvec> &xyz    = sender_patch.get_field<Tvec>(ixyz);
                        PatchDataField<Tscal> &hpart = sender_patch.get_field<Tscal>(ihpart);

                        auto brecv = patch_boxes.values.get(receiver);

                        auto test_volume
                            = paving_function.f_aabb_inv(brecv, infos.xoff, infos.yoff, infos.zoff);
                        auto volume_hmax = int_range_max.get(receiver);

                        return get_ids_in_ghost(
                            dev_sched,
                            q,
                            xyz.get_buf(),
                            hpart.get_buf(),
                            test_volume,
                            volume_hmax,
                            sender_patch.get_obj_cnt());
                    });
        };

        if (shear_x.has_value()) {
            auto paving = get_paving_with_shear(mode, sim_box, shear_x.value());
            run_id_find(paving);
        } else {
            auto paving = get_paving_with_no_shear(mode, sim_box);
            run_id_find(paving);
        }
    }

} // namespace shammodels::sph::modules

using namespace shammodels::sph;

template<class vec>
auto BasicSPHGhostHandler<vec>::find_interfaces(
    SerialPatchTree<vec> &sptree,
    shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
    shamrock::patch::PatchField<flt> &int_range_max) -> GeneratorMap {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shammath;

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    shamrock::patch::SimulationBoxInfo &sim_box = sched.get_sim_box();

    PatchCoordTransform<vec> patch_coord_transf = sim_box.get_patch_transform<vec>();
    vec bsize                                   = sim_box.get_bounding_box_size<vec>();

    GeneratorMap interf_map;

    using CfgClass = sph::BasicSPHGhostHandlerConfig<vec>;
    using BCConfig = typename CfgClass::Variant;

    using BCFree             = typename CfgClass::Free;
    using BCPeriodic         = typename CfgClass::Periodic;
    using BCShearingPeriodic = typename CfgClass::ShearingPeriodic;

    if (BCPeriodic *cfg = std::get_if<BCPeriodic>(&ghost_config)) {
        sycl::host_accessor acc_tf{
            shambase::get_check_ref(int_range_max_tree.internal_buf), sycl::read_only};

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    // sender translation
                    vec periodic_offset = vec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<vec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

                        flt sender_volume = sender_bsize.get_volume();

                        flt sender_h_max = int_range_max.get(psender.id_patch);

                        using PtNode = typename SerialPatchTree<vec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                flt receiv_h_max = acc_tf[tree_id];
                                CoordRange<vec> receiv_exp{
                                    n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                                return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                            },
                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0)
                                    && (zoff == 0)) {
                                    return;
                                }

                                CoordRange<vec> receiv_exp
                                    = CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                                        int_range_max.get(id_found));

                                CoordRange<vec> interf_volume = sender_bsize.get_intersect(
                                    receiv_exp.add_offset(-periodic_offset));

                                interf_map.add_obj(
                                    psender.id_patch,
                                    id_found,
                                    {periodic_offset,
                                     {0, 0, 0},
                                     {xoff, yoff, zoff},
                                     interf_volume,
                                     interf_volume.get_volume() / sender_volume});
                            });
                    });
                }
            }
        }
    } else if (BCShearingPeriodic *cfg = std::get_if<BCShearingPeriodic>(&ghost_config)) {
        sycl::host_accessor acc_tf{
            shambase::get_check_ref(int_range_max_tree.internal_buf), sycl::read_only};

        for_each_patch_shift<flt>(*cfg, bsize, [&](i32_3 ioff, ShiftInfo<flt> shift) {
            i32 xoff = ioff.x();
            i32 yoff = ioff.y();
            i32 zoff = ioff.z();

            vec offset = shift.shift;

            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<vec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
                CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(offset);

                flt sender_volume = sender_bsize.get_volume();

                flt sender_h_max = int_range_max.get(psender.id_patch);

                using PtNode = typename SerialPatchTree<vec>::PtNode;

                sptree.host_for_each_leafs(
                    [&](u64 tree_id, PtNode n) {
                        flt receiv_h_max = acc_tf[tree_id];
                        CoordRange<vec> receiv_exp{
                            n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                        return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                    },
                    [&](u64 id_found, PtNode n) {
                        if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0)
                            && (zoff == 0)) {
                            return;
                        }

                        CoordRange<vec> receiv_exp
                            = CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                                int_range_max.get(id_found));

                        CoordRange<vec> interf_volume
                            = sender_bsize.get_intersect(receiv_exp.add_offset(-offset));

                        interf_map.add_obj(
                            psender.id_patch,
                            id_found,
                            {offset,
                             shift.shift_speed,
                             {xoff, yoff, zoff},
                             interf_volume,
                             interf_volume.get_volume() / sender_volume});

                        // logger::raw_ln("found :",offset, shift.shift_speed, vec{xoff, yoff,
                        // zoff});
                    });
            });
        });

    } else {
        sycl::host_accessor acc_tf{
            shambase::get_check_ref(int_range_max_tree.internal_buf), sycl::read_only};
        // sender translation
        vec periodic_offset = vec{0, 0, 0};

        sched.for_each_local_patch([&](const Patch psender) {
            CoordRange<vec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
            CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

            flt sender_volume = sender_bsize.get_volume();

            flt sender_h_max = int_range_max.get(psender.id_patch);

            using PtNode = typename SerialPatchTree<vec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    flt receiv_h_max = acc_tf[tree_id];
                    CoordRange<vec> receiv_exp{n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                    return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                },
                [&](u64 id_found, PtNode n) {
                    if (id_found == psender.id_patch) {
                        return;
                    }

                    CoordRange<vec> receiv_exp = CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                        int_range_max.get(id_found));

                    CoordRange<vec> interf_volume
                        = sender_bsize.get_intersect(receiv_exp.add_offset(-periodic_offset));

                    interf_map.add_obj(
                        psender.id_patch,
                        id_found,
                        {periodic_offset,
                         {0, 0, 0},
                         {0, 0, 0},
                         interf_volume,
                         interf_volume.get_volume() / sender_volume});
                });
        });
    }

    // interf_map.for_each([](u64 sender, u64 receiver, InterfaceBuildInfos build){
    //     logger::raw_ln("found interface
    //     :",sender,"->",receiver,"ratio:",build.volume_ratio,
    //     "volume:",build.cut_volume.lower,build.cut_volume.upper);
    // });

    return interf_map;
}

template<class vec>
auto BasicSPHGhostHandler<vec>::gen_id_table_interfaces(GeneratorMap &&gen)
    -> shambase::DistributedDataShared<InterfaceIdTable> {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    shambase::DistributedDataShared<InterfaceIdTable> res;

    std::map<u64, f64> send_count_stats;

    gen.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchDataLayer &src = sched.patch_data.get_pdat(sender);
        PatchDataField<vec> &xyz             = src.get_field<vec>(0);

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
            warn_log += shambase::format("\n    patch {} high interf/patch volume: {}", k, v);
            has_warn = true;
        }
    }

    if (has_warn && shamcomm::world_rank() == 0) {
        warn_log = "\n    This can lead to high mpi "
                   "overhead, try to increase the patch split crit"
                   + warn_log;
    }

    if (has_warn) {
        logger::warn_ln("InterfaceGen", "High interface/patch volume ratio." + warn_log);
    }

    return res;
}

template<class vec>
void BasicSPHGhostHandler<vec>::gen_debug_patch_ghost(
    shambase::DistributedDataShared<InterfaceIdTable> &interf_info) {
    StackEntry stack_loc{};

    static u32 cnt_dump_debug = 0;

    std::string loc_graph = "";
    interf_info.for_each([&loc_graph](u64 send, u64 recv, InterfaceIdTable &info) {
        loc_graph += shambase::format("    p{} -> p{}\n", send, recv);
    });

    sched.for_each_patch_data(
        [&](u64 id, shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            if (pdat.get_obj_cnt() > 0) {
                loc_graph += shambase::format(
                    "    p{} [label= \"id={} N={}\"]\n", id, id, pdat.get_obj_cnt());
            }
        });

    std::string dot_graph = "";
    shamcomm::gather_str(loc_graph, dot_graph);

    dot_graph = "strict digraph {\n" + dot_graph + "}";

    if (shamcomm::world_rank() == 0) {
        std::string fname = shambase::format("ghost_graph_{}.dot", cnt_dump_debug);
        logger::info_ln("SPH Ghost", "writing", fname);
        shambase::write_string_to_file(fname, dot_graph);
        cnt_dump_debug++;
    }
}

template class shammodels::sph::BasicSPHGhostHandler<f64_3>;

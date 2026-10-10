// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindGhostInterfaces.cpp
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

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/math.hpp"
#include "shammodels/sph/modules/FindGhostInterfaces.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include <functional>
#include <utility>
#include <vector>

namespace {

    template<class T>
    struct ShiftInfo {
        sycl::vec<T, 3> shift;
        sycl::vec<T, 3> shift_speed;
    };

    template<class T>
    struct ShearPeriodicInfo {
        i32_3 shear_base;
        i32_3 shear_dir;
        T shear_value;
        T shear_speed;
    };

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
        const std::function<void(i32_3, ShiftInfo<T>)> &funct) {

        std::vector<i32_3> list_possible;

        // a bit of dirty fix doesn't hurt
        // this should be done in a better way a some point
        i32 repetition_x = 1 + sham::abs(shearinfo.shear_dir.x());
        i32 repetition_y = 1 + sham::abs(shearinfo.shear_dir.y());
        i32 repetition_z = 1 + sham::abs(shearinfo.shear_dir.z());

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

        for (i32_3 off : list_possible) {
            funct(off, compute_shift_infos(off, shearinfo, bsize));
        }
    }

} // namespace

template<class Tvec>
void shammodels::sph::modules::FindGhostInterfacesFree<Tvec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    using namespace shammath;

    auto edges = get_edges();

    SerialPatchTree<Tvec> &sptree                         = edges.patch_tree.get_patch_tree();
    const shambase::DistributedData<Tscal> &int_range_max = edges.interact_radius.values;

    std::vector<std::pair<u64, CoordRange<Tvec>>> senders;
    edges.local_patch_boxes.values.for_each([&](u64 id, const CoordRange<Tvec> &box) {
        senders.push_back({id, box});
    });

    shambase::DistributedDataShared<InterfaceBuildInfos> &interf_map = edges.interface_infos.values;
    interf_map                                                       = {};

    sycl::host_accessor acc_tf{edges.interact_radius_tree.get_buf(), sycl::read_only};
    // sender translation
    Tvec periodic_offset = Tvec{0, 0, 0};

    sycl::host_accessor tree{shambase::get_check_ref(sptree.serial_tree_buf), sycl::read_only};
    sycl::host_accessor lpid{shambase::get_check_ref(sptree.linked_patch_ids_buf), sycl::read_only};

#pragma omp parallel for
    for (u32 i = 0; i < senders.size(); i++) {
        u64 sender_id                 = senders[i].first;
        CoordRange<Tvec> sender_bsize = senders[i].second;

        CoordRange<Tvec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

        Tscal sender_volume = sender_bsize.get_volume();

        using PtNode = typename SerialPatchTree<Tvec>::PtNode;

        sptree.host_for_each_leafs_internal(
            [&](u64 tree_id, PtNode n) {
                Tscal receiv_h_max = acc_tf[tree_id];
                CoordRange<Tvec> receiv_exp{n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
            },
            [&](u64 id_found, PtNode n) {
                if (id_found == sender_id) {
                    return;
                }

                CoordRange<Tvec> receiv_exp = CoordRange<Tvec>{n.box_min, n.box_max}.expand_all(
                    int_range_max.get(id_found));

                CoordRange<Tvec> interf_volume
                    = sender_bsize.get_intersect(receiv_exp.add_offset(-periodic_offset));

#pragma omp critical
                interf_map.add_obj(
                    sender_id,
                    id_found,
                    {periodic_offset,
                     {0, 0, 0},
                     {0, 0, 0},
                     interf_volume,
                     interf_volume.get_volume() / sender_volume});
            },
            tree,
            lpid);
    }
}

template<class Tvec>
void shammodels::sph::modules::FindGhostInterfacesPeriodic<Tvec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    using namespace shammath;

    auto edges = get_edges();

    const shammath::AABB<Tvec> &sim_box = edges.sim_box.value;
    Tvec bsize                          = sim_box.upper - sim_box.lower;

    SerialPatchTree<Tvec> &sptree                         = edges.patch_tree.get_patch_tree();
    const shambase::DistributedData<Tscal> &int_range_max = edges.interact_radius.values;

    std::vector<std::pair<u64, CoordRange<Tvec>>> senders;
    edges.local_patch_boxes.values.for_each([&](u64 id, const CoordRange<Tvec> &box) {
        senders.push_back({id, box});
    });

    shambase::DistributedDataShared<InterfaceBuildInfos> &interf_map = edges.interface_infos.values;
    interf_map                                                       = {};

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    sycl::host_accessor acc_tf{edges.interact_radius_tree.get_buf(), sycl::read_only};

    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                // sender translation
                Tvec periodic_offset = Tvec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                sycl::host_accessor tree{
                    shambase::get_check_ref(sptree.serial_tree_buf), sycl::read_only};
                sycl::host_accessor lpid{
                    shambase::get_check_ref(sptree.linked_patch_ids_buf), sycl::read_only};

#pragma omp parallel for
                for (u32 i = 0; i < senders.size(); i++) {
                    u64 sender_id                 = senders[i].first;
                    CoordRange<Tvec> sender_bsize = senders[i].second;

                    CoordRange<Tvec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

                    Tscal sender_volume = sender_bsize.get_volume();

                    using PtNode = typename SerialPatchTree<Tvec>::PtNode;

                    sptree.host_for_each_leafs_internal(
                        [&](u64 tree_id, PtNode n) {
                            Tscal receiv_h_max = acc_tf[tree_id];
                            CoordRange<Tvec> receiv_exp{
                                n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                            return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                        },
                        [&](u64 id_found, PtNode n) {
                            if ((id_found == sender_id) && (xoff == 0) && (yoff == 0)
                                && (zoff == 0)) {
                                return;
                            }

                            CoordRange<Tvec> receiv_exp
                                = CoordRange<Tvec>{n.box_min, n.box_max}.expand_all(
                                    int_range_max.get(id_found));

                            CoordRange<Tvec> interf_volume = sender_bsize.get_intersect(
                                receiv_exp.add_offset(-periodic_offset));

#pragma omp critical
                            interf_map.add_obj(
                                sender_id,
                                id_found,
                                {periodic_offset,
                                 {0, 0, 0},
                                 {xoff, yoff, zoff},
                                 interf_volume,
                                 interf_volume.get_volume() / sender_volume});
                        },
                        tree,
                        lpid);
                }
            }
        }
    }
}

template<class Tvec>
void shammodels::sph::modules::FindGhostInterfacesShearingPeriodic<
    Tvec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    using namespace shammath;

    auto edges = get_edges();

    const shammath::AABB<Tvec> &sim_box = edges.sim_box.value;
    Tvec bsize                          = sim_box.upper - sim_box.lower;

    ShearPeriodicInfo<Tscal> shear_info{
        shear_base, shear_dir, shear_speed * edges.time.data, shear_speed};

    SerialPatchTree<Tvec> &sptree                         = edges.patch_tree.get_patch_tree();
    const shambase::DistributedData<Tscal> &int_range_max = edges.interact_radius.values;

    std::vector<std::pair<u64, CoordRange<Tvec>>> senders;
    edges.local_patch_boxes.values.for_each([&](u64 id, const CoordRange<Tvec> &box) {
        senders.push_back({id, box});
    });

    shambase::DistributedDataShared<InterfaceBuildInfos> &interf_map = edges.interface_infos.values;
    interf_map                                                       = {};

    sycl::host_accessor acc_tf{edges.interact_radius_tree.get_buf(), sycl::read_only};

    for_each_patch_shift<Tscal>(shear_info, bsize, [&](i32_3 ioff, ShiftInfo<Tscal> shift) {
        i32 xoff = ioff.x();
        i32 yoff = ioff.y();
        i32 zoff = ioff.z();

        Tvec offset = shift.shift;

        sycl::host_accessor tree{shambase::get_check_ref(sptree.serial_tree_buf), sycl::read_only};
        sycl::host_accessor lpid{
            shambase::get_check_ref(sptree.linked_patch_ids_buf), sycl::read_only};

#pragma omp parallel for
        for (u32 i = 0; i < senders.size(); i++) {
            u64 sender_id                 = senders[i].first;
            CoordRange<Tvec> sender_bsize = senders[i].second;

            CoordRange<Tvec> sender_bsize_off = sender_bsize.add_offset(offset);

            Tscal sender_volume = sender_bsize.get_volume();

            using PtNode = typename SerialPatchTree<Tvec>::PtNode;

            sptree.host_for_each_leafs_internal(
                [&](u64 tree_id, PtNode n) {
                    Tscal receiv_h_max = acc_tf[tree_id];
                    CoordRange<Tvec> receiv_exp{n.box_min - receiv_h_max, n.box_max + receiv_h_max};

                    return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                },
                [&](u64 id_found, PtNode n) {
                    if ((id_found == sender_id) && (xoff == 0) && (yoff == 0) && (zoff == 0)) {
                        return;
                    }

                    CoordRange<Tvec> receiv_exp = CoordRange<Tvec>{n.box_min, n.box_max}.expand_all(
                        int_range_max.get(id_found));

                    CoordRange<Tvec> interf_volume
                        = sender_bsize.get_intersect(receiv_exp.add_offset(-offset));

#pragma omp critical
                    interf_map.add_obj(
                        sender_id,
                        id_found,
                        {offset,
                         shift.shift_speed,
                         {xoff, yoff, zoff},
                         interf_volume,
                         interf_volume.get_volume() / sender_volume});
                },
                tree,
                lpid);
        }
    });
}

template<class Tvec>
std::string shammodels::sph::modules::FindGhostInterfacesFree<Tvec>::_impl_get_tex() const {
    return "TODO";
}

template<class Tvec>
std::string shammodels::sph::modules::FindGhostInterfacesPeriodic<Tvec>::_impl_get_tex() const {
    return "TODO";
}

template<class Tvec>
std::string shammodels::sph::modules::FindGhostInterfacesShearingPeriodic<Tvec>::_impl_get_tex()
    const {
    return "TODO";
}

template class shammodels::sph::modules::FindGhostInterfacesFree<f64_3>;
template class shammodels::sph::modules::FindGhostInterfacesPeriodic<f64_3>;
template class shammodels::sph::modules::FindGhostInterfacesShearingPeriodic<f64_3>;

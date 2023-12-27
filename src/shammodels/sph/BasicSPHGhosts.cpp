// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BasicSPHGhosts.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

/* 

Test code for godbolt


#include <vector>
#include <iostream>

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





#include "BasicSPHGhosts.hpp"
#include "shambase/exception.hpp"
#include <functional>
#include <vector>

template<class T>
struct ShiftInfo{
    sycl::vec<T,3> shift;
    sycl::vec<T,3> shift_speed;
};

template<class T>
using ShearPeriodicInfo = typename shammodels::sph::BasicSPHGhostHandlerConfig<sycl::vec<T,3>>::ShearingPeriodic;


template<class T>
inline ShiftInfo<T> compute_shift_infos(
    i32_3 ioff, ShearPeriodicInfo<T> shear, sycl::vec<T,3> bsize
    ){

    i32 dx = ioff.x()*shear.shear_base.x();
    i32 dy = ioff.y()*shear.shear_base.y();
    i32 dz = ioff.z()*shear.shear_base.z();

    i32 d = dx + dy + dz;

    sycl::vec<T,3> shift = {
        (d*shear.shear_dir.x())*shear.shear_value + bsize.x()*ioff.x(),
        (d*shear.shear_dir.y())*shear.shear_value + bsize.y()*ioff.y() ,
        (d*shear.shear_dir.z())*shear.shear_value + bsize.z()*ioff.z()
    };
    sycl::vec<T,3> shift_speed = {
        (d*shear.shear_dir.x())*shear.shear_speed,
        (d*shear.shear_dir.y())*shear.shear_speed,
        (d*shear.shear_dir.z())*shear.shear_speed
    };

    return {shift,shift_speed};
}

template<class T>
inline void for_each_patch_shift(ShearPeriodicInfo<T> shearinfo, sycl::vec<T,3> bsize, std::function<void(i32_3, ShiftInfo<T>)> funct){

    i32_3 loop_offset = {0,0,0};

    std::vector<i32_3> list_possible;

    //logger::raw_ln("testing :",shearinfo.shear_value,shearinfo.shear_dir, shearinfo.shear_base);

    // a bit of dirty fix doesn't hurt
    // this should be done in a better way a some point
    i32 repetition_x = 1 + abs(shearinfo.shear_dir.x());
    i32 repetition_y = 1 + abs(shearinfo.shear_dir.y());
    i32 repetition_z = 1 + abs(shearinfo.shear_dir.z());

    T sz = bsize.x()*shearinfo.shear_dir.x() + bsize.y()*shearinfo.shear_dir.y() + bsize.z()*shearinfo.shear_dir.z();

    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {
                
                
                i32 dx = xoff*shearinfo.shear_base.x();
                i32 dy = yoff*shearinfo.shear_base.y();
                i32 dz = zoff*shearinfo.shear_base.z();

                i32 d = dx + dy + dz;

                i32 df = -int(d * shearinfo.shear_value/sz);
                
                i32_3 off_d = {
                    shearinfo.shear_dir.x()*df,
                    shearinfo.shear_dir.y()*df,
                    shearinfo.shear_dir.z()*df
                };
                
                list_possible.push_back({xoff+off_d.x(),yoff+off_d.y(),zoff+off_d.z()});
            }
        }
    }

    //logger::raw_ln("trying", list_possible.size(), "patches ghosts");

    for(i32_3 off : list_possible){

        auto shift = compute_shift_infos(off,shearinfo,bsize);

        //logger::raw_ln("check :",off,shift.shift, shift.shift_speed);

        funct(off,shift);
        
    }
    


}





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

    shamrock::patch::SimulationBoxInfo & sim_box = sched.get_sim_box();

    PatchCoordTransform<vec> patch_coord_transf = sim_box.get_patch_transform<vec>();
    vec bsize                                   = sim_box.get_bounding_box_size<vec>();

    GeneratorMap interf_map;


    using CfgClass = sph::BasicSPHGhostHandlerConfig<vec>;
    using BCConfig = typename CfgClass::Variant;

    using BCFree = typename CfgClass::Free;
    using BCPeriodic = typename CfgClass::Periodic;
    using BCShearingPeriodic = typename CfgClass::ShearingPeriodic;


    if(BCPeriodic * cfg = std::get_if<BCPeriodic>(&ghost_config)){
        sycl::host_accessor acc_tf{shambase::get_check_ref(int_range_max_tree.internal_buf),
                                   sycl::read_only};

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
                                CoordRange<vec> receiv_exp{n.box_min - receiv_h_max,
                                                           n.box_max + receiv_h_max};

                                return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                            },
                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                                    (zoff == 0)) {
                                    return;
                                }

                                CoordRange<vec> receiv_exp =
                                    CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                                        int_range_max.get(id_found));

                                CoordRange<vec> interf_volume = sender_bsize.get_intersect(
                                    receiv_exp.add_offset(-periodic_offset));

                                interf_map.add_obj(psender.id_patch,
                                                   id_found,
                                                   {periodic_offset,{0,0,0},
                                                    {xoff, yoff, zoff},
                                                    interf_volume,
                                                    interf_volume.get_volume() / sender_volume});
                            });
                    });
                }
            }
        }
    }else if(BCShearingPeriodic * cfg = std::get_if<BCShearingPeriodic>(&ghost_config)){
        sycl::host_accessor acc_tf{shambase::get_check_ref(int_range_max_tree.internal_buf),
                                   sycl::read_only};

        for_each_patch_shift<flt>(*cfg, bsize, [&](i32_3 ioff, ShiftInfo<flt> shift){
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
                        CoordRange<vec> receiv_exp{n.box_min - receiv_h_max,
                                                    n.box_max + receiv_h_max};

                        return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                    },
                    [&](u64 id_found, PtNode n) {
                        if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                            (zoff == 0)) {
                            return;
                        }

                        CoordRange<vec> receiv_exp =
                            CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                                int_range_max.get(id_found));

                        CoordRange<vec> interf_volume = sender_bsize.get_intersect(
                            receiv_exp.add_offset(-offset));

                        interf_map.add_obj(psender.id_patch,
                                            id_found,
                                            {offset,shift.shift_speed,
                                            {xoff, yoff, zoff},
                                            interf_volume,
                                            interf_volume.get_volume() / sender_volume});

                        //logger::raw_ln("found :",offset, shift.shift_speed, vec{xoff, yoff, zoff});

                    });
            });
        });

                    
        
    }else{
        sycl::host_accessor acc_tf{shambase::get_check_ref(int_range_max_tree.internal_buf),
                                   sycl::read_only};
        // sender translation
        vec periodic_offset = vec{0,0,0};

        sched.for_each_local_patch([&](const Patch psender) {
            CoordRange<vec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
            CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

            flt sender_volume = sender_bsize.get_volume();

            flt sender_h_max = int_range_max.get(psender.id_patch);

            using PtNode = typename SerialPatchTree<vec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    flt receiv_h_max = acc_tf[tree_id];
                    CoordRange<vec> receiv_exp{n.box_min - receiv_h_max,
                                                n.box_max + receiv_h_max};

                    return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                },
                [&](u64 id_found, PtNode n) {
                    if (id_found == psender.id_patch) {
                        return;
                    }

                    CoordRange<vec> receiv_exp =
                        CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                            int_range_max.get(id_found));

                    CoordRange<vec> interf_volume = sender_bsize.get_intersect(
                        receiv_exp.add_offset(-periodic_offset));

                    interf_map.add_obj(psender.id_patch,
                                        id_found,
                                        {periodic_offset,{0,0,0},
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
        shamrock::patch::PatchData &src = sched.patch_data.get_pdat(sender);
        PatchDataField<vec> &xyz        = src.get_field<vec>(0);

        std::unique_ptr<sycl::buffer<u32>> idxs = xyz.get_elements_with_range_buf(
            [&](vec val, vec vmin, vec vmax) {
                return Patch::is_in_patch_converted(val, vmin, vmax);
            },
            build.cut_volume.lower,
            build.cut_volume.upper);

        u32 pcnt = 0;
        if (bool(idxs)) {
            pcnt = idxs->size();
        }

        // prevent sending empty patches
        if (pcnt == 0) {
            return;
        }

        f64 ratio = f64(pcnt) / f64(src.get_obj_cnt());

        logger::debug_sycl_ln("InterfaceGen",
                              "gen interface :",
                              sender,
                              "->",
                              receiver,
                              "volume ratio:",
                              build.volume_ratio,
                              "part_ratio:",
                              ratio);

        res.add_obj(sender, receiver, InterfaceIdTable{build, std::move(idxs), ratio});

        send_count_stats[sender] += ratio;
    });

    bool has_warn = false;

    for (auto &[k, v] : send_count_stats) {
        if (v > 0.2) {
            logger::warn_ln("InterfaceGen", "patch", k, " high ratio volume/interf:", v);
            has_warn = true;
        }
    }

    if (has_warn) {
        logger::warn_ln("InterfaceGen",
                        "the ratio patch/interface is high, which can lead to high mpi "
                        "overhead, try incresing the patch split crit");
    }

    return res;
}

template class shammodels::sph::BasicSPHGhostHandler<f64_3>;
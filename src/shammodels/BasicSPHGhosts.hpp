// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "shambase/DistributedData.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/BasicGas.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"

namespace shammodels::sph {

    template<class vec>
    class BasicGasPeriodicGhostHandler{

        PatchScheduler & sched;

        public:

        using flt      = shambase::VecComponent<vec>;

        struct InterfaceBuildInfos{
            vec offset;
            shammath::CoordRange<vec> cut_volume;
            flt volume_ratio;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;

        BasicGasPeriodicGhostHandler(PatchScheduler & sched) : sched(sched){}

        GeneratorMap find_interfaces(
            SerialPatchTree<vec> & sptree,
            shamrock::patch::PatchtreeField<flt> & int_range_max_tree,
            shamrock::patch::PatchField<flt> & int_range_max){

            using namespace shamrock::patch;
            using namespace shammath;

            i32 repetition_x = 1;
            i32 repetition_y = 1;
            i32 repetition_z = 1;
            
            PatchCoordTransform<vec> patch_coord_transf = sched.get_sim_box().get_patch_transform<vec>();
            vec bsize = sched.get_sim_box().get_bounding_box_size<vec>();

            GeneratorMap interf_map;

            {
                sycl::host_accessor acc_tf {shambase::get_check_ref(int_range_max_tree.internal_buf), sycl::read_only};
            

                for(i32 xoff = - repetition_x; xoff <= repetition_x; xoff ++){
                    for(i32 yoff = - repetition_y; yoff <= repetition_y; yoff ++){
                        for(i32 zoff = - repetition_z; zoff <= repetition_z; zoff ++){


                            // sender translation
                            vec periodic_offset = vec{
                                xoff*bsize.x(),
                                yoff*bsize.y(),
                                zoff*bsize.z()
                                };
                            
                            sched.for_each_local_patch([&](const Patch psender){
                                        
                                CoordRange<vec> sender_bsize = patch_coord_transf.to_obj_coord(psender);
                                CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

                                flt sender_volume = sender_bsize.get_volume();

                                flt sender_h_max = int_range_max.get(psender.id_patch);

                                using PtNode = typename SerialPatchTree<vec>::PtNode;

                                sptree.host_for_each_leafs([&](u64 tree_id, PtNode n){
                                    flt receiv_h_max = acc_tf[tree_id];
                                    CoordRange<vec> receiv_exp {n.box_min-receiv_h_max,n.box_max+receiv_h_max};
                                    
                                    return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                                }, [&](u64 id_found, PtNode n){
                                    if((id_found == psender.id_patch) && 
                                    (xoff == 0) && 
                                    (yoff == 0) && 
                                    (zoff == 0)){return;}

                                    CoordRange<vec> receiv_exp =
                                        CoordRange<vec>{n.box_min,n.box_max}
                                        .expand_all(int_range_max.get(id_found));

                                    CoordRange<vec> interf_volume = sender_bsize
                                        .get_intersect(receiv_exp.add_offset(-periodic_offset));

                                    interf_map.add_obj(psender.id_patch, id_found, {
                                        periodic_offset,
                                        interf_volume,
                                        interf_volume.get_volume()/sender_volume
                                    });
                                });
                                
                            });

                        }
                    }
                }
            }

            //interf_map.for_each([](u64 sender, u64 receiver, InterfaceBuildInfos build){
            //    logger::raw_ln("found interface :",sender,"->",receiver,"ratio:",build.volume_ratio, "volume:",build.cut_volume.lower,build.cut_volume.upper);
            //});

            return interf_map;
        }
        

        struct InterfaceIdTable{
            InterfaceBuildInfos build_infos;
            std::unique_ptr<sycl::buffer<u32>> ids_interf;
            f64 part_cnt_ratio;
        };

        shambase::DistributedDataShared<InterfaceIdTable> gen_id_table_interfaces(GeneratorMap && gen){
            using namespace shamrock::patch;


            shambase::DistributedDataShared<InterfaceIdTable> res;

            std::map<u64, f64> send_count_stats;

            gen.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build){
                shamrock::patch::PatchData & src = sched.patch_data.get_pdat(sender);
                PatchDataField<vec> & xyz = src.get_field<vec>(0);
                
                std::unique_ptr<sycl::buffer<u32>> idxs = xyz.get_elements_with_range_buf(
                        [&](vec val,vec vmin, vec vmax){
                            return Patch::is_in_patch_converted(val, vmin,vmax);
                        },
                        build.cut_volume.lower,build.cut_volume.upper
                    );
                
                u32 pcnt = 0;
                if(bool(idxs)){
                    pcnt = idxs->size();
                }

                //prevent sending empty patches
                if(pcnt == 0){return;}

                f64 ratio = f64(pcnt)/f64(src.get_obj_cnt());

                logger::debug_sycl_ln("InterfaceGen","gen interface :",sender,"->",receiver,"volume ratio:",build.volume_ratio, "part_ratio:",ratio);

                res.add_obj(sender, receiver, InterfaceIdTable{
                    build,
                    std::move(idxs),
                    ratio
                });

                send_count_stats[sender] += ratio;

            });

            bool has_warn = false;

            for(auto & [k,v] : send_count_stats){
                if(v > 0.2){
                    logger::warn_ln("InterfaceGen", "patch",k," high ratio volume/interf:",v);
                    has_warn = true;
                }
            }

            if(has_warn){
                logger::warn_ln("InterfaceGen", "the ratio patch/interface is high, which can lead to high mpi overhead, try incresing the patch split crit");
            }

            return res;
        }
        
        shambase::DistributedDataShared<InterfaceIdTable> make_interface_cache(
            SerialPatchTree<vec> & sptree,
            shamrock::patch::PatchtreeField<flt> & int_range_max_tree,
            shamrock::patch::PatchField<flt> & int_range_max){

            return gen_id_table_interfaces(find_interfaces(sptree, int_range_max_tree, int_range_max));
        }


    };

        

}
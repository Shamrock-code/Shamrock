// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "basic_sph_gas.hpp"
#include "aliases.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shammodels/generic/algs/cfl_utils.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shammodels/generic/algs/integrators_utils.hpp"
#include "shamrock/sph/forces.hpp"

#include <array>
#include <memory>
#include <string>

//%Impl status : Good


const std::string console_tag = "[BasicSPHGas] ";


template<class flt, class Kernel> 
void models::sph::BasicSPHGas<flt,Kernel>::check_valid(){
    if (cfl_cour < 0) {
        throw ShamAPIException(console_tag + "cfl courant not set");
    }

    if (cfl_force < 0) {
        throw ShamAPIException(console_tag + "cfl force not set");
    }

    if (gpart_mass < 0) {
        throw ShamAPIException(console_tag + "particle mass not set");
    }
}



template<class flt, class Kernel> 
void models::sph::BasicSPHGas<flt,Kernel>::init(){

}

template<class flt, class Kernel> 
f64 models::sph::BasicSPHGas<flt,Kernel>::evolve(PatchScheduler &sched, f64 old_time, f64 target_time){

    using namespace shamrock::patch;

    check_valid();

    logger::info_ln("BasicSPHGas", "evolve t=",old_time);


    Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);


    const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
    const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
    const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
    const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

    const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

    PatchComputeField<f32> pressure_field;


    f64 step_time = stepper.step(old_time, true, true, 

        [&](u64  /*id_patch*/, PatchData &pdat) {
        
            flt cfl_val = CflUtility<flt>::basic_cfl(pdat, [&](sycl::handler &cgh, sycl::buffer<flt> & buf_cfl, sycl::range<1> range_it){

                auto arr = buf_cfl.template get_access<sycl::access::mode::discard_write>(cgh);


                auto acc_hpart = pdat.get_field<flt>(ihpart).get_buf()->template get_access<sycl::access::mode::read>(cgh);
                auto acc_axyz  = pdat.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read>(cgh);

                const flt cs = eos_cs;

                const flt c_cour  = cfl_cour;
                const flt c_force = cfl_force;

                cgh.parallel_for(range_it, [=](sycl::item<1> item) {
                    u32 i = (u32)item.get_id(0);

                    flt h_a    = acc_hpart[item];
                    vec3 axyz = acc_axyz[item];

                    flt dtcfl_p = c_cour * h_a / cs;
                    flt dtcfl_a = c_force * sycl::sqrt(h_a / sycl::length(axyz));

                    arr[i] = sycl::min(dtcfl_p, dtcfl_a);
                });

            });

            logger::info_ln("BasicSPHGas", "cfl dt :",cfl_val);

            f32 cfl_dt_loc = cfl_val;//sycl::min(f32(0.001),cfl_val);

            if(cfl_dt_loc + old_time > target_time){
                cfl_dt_loc = target_time - old_time;
            }

            return cfl_dt_loc;

        }, 
        
        [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ,flt hdt){
            
            sycl::buffer<vec3> & vxyz =  * pdat.get_field<vec3>(ivxyz).get_buf();
            sycl::buffer<vec3> & axyz =  * pdat.get_field<vec3>(iaxyz).get_buf();

            field_advance_time(queue, vxyz, axyz, range_npart, hdt);

        }, 
        
        [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ){
            auto ker_predict_step = [&](sycl::handler &cgh) {
                auto acc_axyz = pdat.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
                auto acc_axyz_old = pdat.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

                // Executing kernel
                cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                    vec3 axyz     = acc_axyz[item];
                    vec3 axyz_old = acc_axyz_old[item];

                    acc_axyz[item]     = axyz_old;
                    acc_axyz_old[item] = axyz;

                });
            };

            queue.submit(ker_predict_step);
        },
        [&](PatchScheduler & sched, 
            std::unordered_map<u64, MergedPatchData<flt>>& merge_pdat, 
            std::unordered_map<u64, MergedPatchCompField<flt,flt>>& hnew_field_merged,
            std::unordered_map<u64, MergedPatchCompField<flt,flt>>& omega_field_merged
            ) {


                std::unordered_map<u64, u32> size_map;

                for(auto & [k,buf] : merge_pdat){
                    size_map[k] = buf.data.get_obj_cnt();
                }

                pressure_field.generate(sched,size_map);

                sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                    auto & hnew  = hnew_field_merged[id_patch].buf.get_buf();
                    auto & press  = pressure_field.get_buf(id_patch);

                    sycl::range range_npart{size_map[id_patch]}; //TODO remove ref to size

                    auto cs = eos_cs;
                    auto part_mass = gpart_mass;

                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        auto h = hnew->template get_access<sycl::access::mode::read>(cgh);

                        auto p = press->get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for(range_npart,
                                [=](sycl::item<1> item) { 

                                    using namespace shamrock::sph;
                                    
                                    p[item] =   cs*cs*rho_h(part_mass, h[item],Kernel::hfactd)  ; 
                                    
                                    
                                    });
                    });

                });
            }
        ,
        [&](


            PatchScheduler & sched, 
            std::unordered_map<u64, std::unique_ptr<RadixTree<u_morton, vec3>>>& radix_trees,
            std::unordered_map<u64, std::unique_ptr<RadixTreeField<flt> >> & cell_int_rads,
            std::unordered_map<u64, MergedPatchData<flt>>& merge_pdat, 
            std::unordered_map<u64, MergedPatchCompField<flt,flt>>& hnew_field_merged,
            std::unordered_map<u64, MergedPatchCompField<flt,flt>>& omega_field_merged,
            flt htol_up_tol
            ){
            sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                if (merge_pdat.at(id_patch).or_element_cnt == 0){
                    logger::info_ln("BasicSPHGas","patch id =",id_patch,"is empty => skipping");
                }


                PatchData & pdat_merge = merge_pdat.at(id_patch).data;

                sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf.get_buf();
                sycl::buffer<f32> &omega = *omega_field_merged[id_patch].buf.get_buf();

                auto & press  = pressure_field.get_buf(id_patch);

                sycl::range range_npart{merge_pdat.at(id_patch).or_element_cnt};

            
                logger::info_ln("BasicSPHGas","patch : n°" ,id_patch , "compute forces");

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                    auto omga  = omega.get_access<sycl::access::mode::read>(cgh);

                    auto r        = pdat_merge.get_field<f32_3>(ixyz).get_buf()->get_access<sycl::access::mode::read>(cgh);
                    auto acc_axyz = pdat_merge.get_field<f32_3>(iaxyz).get_buf()->get_access<sycl::access::mode::discard_write>(cgh);

                    using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                    Rta tree_acc(*radix_trees[id_patch], cgh);

                    auto cell_int_r =
                        cell_int_rads.at(id_patch)->radix_tree_field_buf->template get_access<sycl::access::mode::read>(cgh);


                    auto pres = press->get_access<sycl::access::mode::read>(cgh);

                    const f32 part_mass = gpart_mass;
                    //const f32 cs        = eos_cs;

                    const f32 htol = htol_up_tol;
                    using namespace shamrock::sph;
                    // sycl::stream out(65000,65000,cgh);

                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32)item.get_id(0);

                        f32_3 sum_axyz = {0, 0, 0};
                        f32 h_a        = h_new[id_a];

                        f32_3 xyz_a = r[id_a];

                        f32 rho_a    = rho_h(part_mass, h_a,Kernel::hfactd);
                        f32 rho_a_sq = rho_a * rho_a;

                        f32 P_a     = pres[id_a];
                        //f32 P_a     = cs * cs * rho_a;
                        f32 omega_a = omga[id_a];

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc, &xyz_a, &inter_box_a_min, &inter_box_a_max, &cell_int_r,&htol](u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern * htol;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                            },
                            [&](u32 id_b) {
                                // compute only omega_a
                                f32_3 dr = xyz_a - r[id_b];
                                f32 rab  = sycl::length(dr);
                                f32 h_b  = h_new[id_b];

                                if (rab > h_a * Kernel::Rkern && rab > h_b * Kernel::Rkern)
                                    return;

                                f32_3 r_ab_unit = dr / rab;

                                if (rab < 1e-9) {
                                    r_ab_unit = {0, 0, 0};
                                }

                                
                                f32 rho_b   = rho_h(part_mass, h_b,Kernel::hfactd);
                                f32 P_b     = pres[id_b];
                                //f32 P_b     = cs * cs * rho_b;
                                f32 omega_b = omga[id_b];

                                f32_3 tmp = sph_pressure_symetric<f32_3, f32>(part_mass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
                                                                    omega_b, r_ab_unit * Kernel::dW(rab, h_a),
                                                                    r_ab_unit * Kernel::dW(rab, h_b));

                                sum_axyz += tmp;
                            },
                            [](u32 node_id) {});

                        // out << "sum : " << sum_axyz << "\n";

                        acc_axyz[id_a] = sum_axyz;
                    });
                });
                

            });

        }, 
        
        [&](sycl::queue&  queue, PatchData& buf, sycl::range<1> range_npart ,flt hdt){
            

            auto ker_corect_step = [&](sycl::handler &cgh) {
                auto acc_vxyz     = buf.get_field<vec3>(ivxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
                auto acc_axyz     = buf.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
                auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

                // Executing kernel
                cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                    //u32 gid = (u32)item.get_id();
                    //
                    //vec3 &vxyz     = acc_vxyz[item];
                    //vec3 &axyz     = acc_axyz[item];
                    //vec3 &axyz_old = acc_axyz_old[item];

                    // v^* = v^{n + 1/2} + dt/2 a^n
                    acc_vxyz[item] = acc_vxyz[item] + (hdt) * (acc_axyz[item] - acc_axyz_old[item]);
                });
            };

            queue.submit(ker_corect_step);
        });

    return step_time;
}

template<class flt, class Kernel> 
void models::sph::BasicSPHGas<flt,Kernel>::dump(std::string prefix){
    std::cout << "dump : "<< prefix << std::endl;
}

template<class flt, class Kernel> 
void models::sph::BasicSPHGas<flt,Kernel>::restart_dump(std::string prefix){
    std::cout << "restart dump : "<< prefix << std::endl;
}

template<class flt, class Kernel> 
void models::sph::BasicSPHGas<flt,Kernel>::close(){
    
}



template class models::sph::BasicSPHGas<f32,shamrock::sph::kernels::M4<f32>>;
template class models::sph::BasicSPHGas<f32,shamrock::sph::kernels::M6<f32>>;


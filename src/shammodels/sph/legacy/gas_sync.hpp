// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Deprecated


#pragma once

#include "shamrock/legacy/utils/syclreduction.hpp"
#include "aliases.hpp"


#include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shammodels/generic/physics/units.hpp"
#include "shammodels/sph/integrators/leapfrog.hpp"
#include "forces.hpp"
#include "shammodels/generic/algs/integrators_utils.hpp"
#include "shammodels/generic/algs/cfl_utils.hpp"
#include "shammodels/sph/base/sphpart.hpp"
#include <memory>
#include <unordered_map>
#include <vector>



template <class flt> class SPHTimestepperLeapfrogIsotGasSync {
  public:

  constexpr static f32 gpart_mass = 2e-4;

    using vec3 = sycl::vec<flt, 3>;
    using u_morton = u32;
    using Kernel = models::sph::kernels::M4<f32>;

    using Stepper = integrators::sph::LeapfrogGeneral<flt, Kernel, u_morton>;

    struct SyncPart {
        vec3 pos;
        vec3 vel;
        flt mass;
    };

    std::vector<SyncPart> sync_parts;

    inline void step(PatchScheduler &sched, std::string dump_folder, u32 step_cnt, f64 &step_time) {

        bool periodic_bc = true;

        flt htol_up_tol  = 1.4;
        flt htol_up_iter = 1.2;

        const flt cfl_cour  = 0.1;
        const flt cfl_force = 0.1;

        const flt eos_cs = 1;



        Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);
        bool do_force = step_cnt > 4;
        bool do_corrector = step_cnt > 5;






        

        const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

        const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

        PatchComputeField<f32> pressure_field;


        step_time = stepper.step(step_time, do_force, do_corrector, 
        
        [&](u64 id_patch, PatchDataBuffer &pdat_buf) {

            
                flt cfl_val = CflUtility<flt>::basic_cfl(pdat_buf, [&](sycl::handler &cgh, sycl::buffer<flt> & buf_cfl, sycl::range<1> range_it){

                    auto arr = buf_cfl.template get_access<sycl::access::mode::discard_write>(cgh);
                    auto acc_hpart = pdat_buf.get_field<flt>(ihpart)->template get_access<sycl::access::mode::read>(cgh);
                    auto acc_axyz  = pdat_buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read>(cgh);

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

                std::cout << "cfl dt : " << cfl_val << std::endl;

                return sycl::min(f32(0.005),cfl_val);

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ,flt hdt){
                
                sycl::buffer<vec3> & vxyz =  * buf.get_field<vec3>(ivxyz);
                sycl::buffer<vec3> & axyz =  * buf.get_field<vec3>(iaxyz);

                field_advance_time(queue, vxyz, axyz, range_npart, hdt);

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ){
                auto ker_predict_step = [&](sycl::handler &cgh) {
                    auto acc_axyz = buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old)->template get_access<sycl::access::mode::read_write>(cgh);

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
                std::unordered_map<u64, MergedPatchDataBuffer<vec3>>& merge_pdat_buf, 
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& hnew_field_merged,
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& omega_field_merged
                ) {


                    std::unordered_map<u64, u32> size_map;

                    for(auto & [k,buf] : merge_pdat_buf){
                        size_map[k] = buf.data->element_count;
                    }

                    pressure_field.generate(sched,size_map);
                    pressure_field.to_sycl();

                    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                        sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf;
                        sycl::buffer<f32> &press  = *pressure_field.field_data_buf[id_patch];

                        sycl::range range_npart{hnew.size()};

                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                            auto h = hnew.get_access<sycl::access::mode::read>(cgh);

                            auto p = press.get_access<sycl::access::mode::discard_write>(cgh);

                            cgh.parallel_for(range_npart,
                                    [=](sycl::item<1> item) { 
                                        
                                        p[item] =   eos_cs*eos_cs*rho_h(gpart_mass, h[item])  ; 
                                        
                                        
                                        });
                        });

                    });
                }
            ,
            [&](


                PatchScheduler & sched, 
                std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, vec3>>>& radix_trees,
                std::unordered_map<u64, MergedPatchDataBuffer<vec3>>& merge_pdat_buf, 
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& hnew_field_merged,
                std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>>& omega_field_merged,
                flt htol_up_tol
                ){
                sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                    if (merge_pdat_buf.at(id_patch).or_element_cnt == 0){
                        std::cout << " empty => skipping" << std::endl;return;
                    }

                    

                    PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

                    sycl::buffer<f32> &hnew  = *hnew_field_merged[id_patch].buf;
                    sycl::buffer<f32> &omega = *omega_field_merged[id_patch].buf;

                    sycl::buffer<f32> &press  = *pressure_field.field_data_buf[id_patch];

                    sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

                
                    std::cout << "patch : n°" << id_patch << "compute forces" << std::endl;
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        auto h_new = hnew.get_access<sycl::access::mode::read>(cgh);
                        auto omga  = omega.get_access<sycl::access::mode::read>(cgh);

                        auto r        = pdat_buf_merge.get_field<f32_3>(ixyz)->get_access<sycl::access::mode::read>(cgh);
                        auto acc_axyz = pdat_buf_merge.get_field<f32_3>(iaxyz)->get_access<sycl::access::mode::discard_write>(cgh);

                        using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                        Rta tree_acc(*radix_trees[id_patch], cgh);

                        auto cell_int_r =
                            radix_trees[id_patch]->buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);


                        auto pres = press.get_access<sycl::access::mode::read>(cgh);

                        const f32 part_mass = gpart_mass;
                        //const f32 cs        = eos_cs;

                        const f32 htol = htol_up_tol;

                        // sycl::stream out(65000,65000,cgh);

                        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                            u32 id_a = (u32)item.get_id(0);

                            f32_3 sum_axyz = {0, 0, 0};
                            f32 h_a        = h_new[id_a];

                            f32_3 xyz_a = r[id_a];

                            f32 rho_a    = rho_h(part_mass, h_a);
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

                                    f32 rho_b   = rho_h(part_mass, h_b);
                                    f32 P_b     = pres[id_b];
                                    //f32 P_b     = cs * cs * rho_b;
                                    f32 omega_b = omga[id_b];

                                    f32_3 tmp = sph_pressure<f32_3, f32>(part_mass, rho_a_sq, rho_b * rho_b, P_a, P_b, omega_a,
                                                                        omega_b, 0, 0, r_ab_unit * Kernel::dW(rab, h_a),
                                                                        r_ab_unit * Kernel::dW(rab, h_b));

                                    sum_axyz += tmp;
                                },
                                [](u32 node_id) {});

                            // out << "sum : " << sum_axyz << "\n";

                            acc_axyz[id_a] = sum_axyz;
                        });
                    });
                    

                });

                if(sync_parts.size() > 0){
                    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
                        if (merge_pdat_buf.at(id_patch).or_element_cnt == 0){
                            std::cout << " empty => skipping" << std::endl;return;
                        }

                        

                        PatchDataBuffer &pdat_buf_merge = *merge_pdat_buf.at(id_patch).data;

                        sycl::buffer<SyncPart> syncs ( sync_parts.data(), sync_parts.size() );

                        sycl::range range_npart{merge_pdat_buf.at(id_patch).or_element_cnt};

                        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                            auto r        = pdat_buf_merge.get_field<f32_3>(ixyz)->get_access<sycl::access::mode::read>(cgh);
                            auto acc_axyz = pdat_buf_merge.get_field<f32_3>(iaxyz)->get_access<sycl::access::mode::read_write>(cgh);

                            auto acc_synk = syncs.template get_access<sycl::access::mode::read>(cgh);

                            u32 num_sync = sync_parts.size();

                            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                                u32 id_a = (u32)item.get_id(0);

                                f32_3 xyz_a = r[id_a];

                                f32_3 sum_axyz{0,0,0};

                                for(u32 i = 0; i < num_sync; i++){

                                    SyncPart p = acc_synk[i];

                                    vec3 dr = xyz_a - p.pos;

                                    flt val =  units::G_si * gpart_mass * p.mass / (sycl::dot(dr,dr));

                                    if(val > 1e2) {
                                        val = 1e2;
                                    }

                                    sum_axyz -= val * dr / (sycl::length(dr));
                                }

                                acc_axyz[id_a] += sum_axyz;
                            });
                        });
                        

                    });

                }

            }, 
            
            [&](sycl::queue&  queue, PatchDataBuffer& buf, sycl::range<1> range_npart ,flt hdt){
                

                auto ker_corect_step = [&](sycl::handler &cgh) {
                    auto acc_vxyz     = buf.get_field<vec3>(ivxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz     = buf.get_field<vec3>(iaxyz)->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old)->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 gid = (u32)item.get_id();

                        vec3 &vxyz     = acc_vxyz[item];
                        vec3 &axyz     = acc_axyz[item];
                        vec3 &axyz_old = acc_axyz_old[item];

                        // v^* = v^{n + 1/2} + dt/2 a^n
                        vxyz = vxyz + (hdt) * (axyz - axyz_old);
                    });
                };

                queue.submit(ker_corect_step);
            });



    }
};

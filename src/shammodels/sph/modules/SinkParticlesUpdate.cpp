// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/reduction.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
using SinkUpdate = shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>; 

template<class Tvec, template<class> class SPHKernel>
void SinkUpdate<Tvec, SPHKernel>::accrete_particles(Tscal gpart_mass){
    StackEntry stack_loc{};

    if(storage.sinks.is_empty()){
        return;
    }

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("xyz");

    sycl::queue & q = shamsys::instance::get_compute_queue();

    std::vector<Sink> & sink_parts = storage.sinks.get();

    for (Sink & s : sink_parts) {

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sycl::buffer<Tvec> &buf_xyz = pdat.get_field_buf_ref<Tvec>(ixyz);
            sycl::buffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

            sycl::buffer<u32> not_accreted (Nobj);
            sycl::buffer<u32> accreted (Nobj);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor not_acc {not_accreted, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor acc {accreted, cgh, sycl::write_only, sycl::no_init};

                Tvec r_sink = s.pos;
                Tscal acc_rad2 = s.accretion_radius*s.accretion_radius;

                shambase::parralel_for(cgh, Nobj,"check accretion", [=](i32 id_a){
                    Tvec r = xyz[id_a] - r_sink;
                    bool not_accreted = sycl::dot(r,r) > acc_rad2;
                    not_acc[id_a] = (not_accreted) ? 1 : 0;
                    acc[id_a] = (!not_accreted) ? 1 : 0;
                });
            });

            std::tuple<std::optional<sycl::buffer<u32>>, u32> id_list_keep = shamalgs::numeric::stream_compact(
                q, not_accreted, Nobj);

            std::tuple<std::optional<sycl::buffer<u32>>, u32> id_list_accrete = shamalgs::numeric::stream_compact(
                q, accreted, Nobj);

            //sum accreted values onto sink

            if(std::get<1>(id_list_accrete) > 0){

                u32 Naccrete = std::get<1>(id_list_accrete);

                Tscal acc_mass = gpart_mass*Naccrete;

                sycl::buffer<Tvec> pxyz_acc (Naccrete);
                q.submit([&,gpart_mass](sycl::handler &cgh) {
                    sycl::accessor id_acc {*std::get<0>(id_list_accrete), cgh, sycl::read_only};
                    sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};

                    sycl::accessor accretion_p {pxyz_acc, cgh, sycl::write_only};

                    shambase::parralel_for(cgh, Naccrete,"compute sum momentum accretion", [=](i32 id_a){
                        accretion_p[id_a] = gpart_mass * vxyz[id_acc[id_a]];
                    });
                });

                Tvec acc_pxyz = shamalgs::reduction::sum(q, pxyz_acc, 0, Naccrete);

                s.mass += acc_mass;
                s.velocity += acc_pxyz/s.mass;

                logger::raw_ln("accretion : += ",acc_mass, acc_pxyz/s.mass);

                pdat.keep_ids(*std::get<0>(id_list_keep), std::get<1>(id_list_keep));
            }


  
        });


        
    }

}

template<class Tvec, template<class> class SPHKernel>
void SinkUpdate<Tvec, SPHKernel>::predictor_step(Tscal dt){

    StackEntry stack_loc{};

    if(storage.sinks.is_empty()){
        return;
    }

    compute_ext_forces();
    
    std::vector<Sink> & sink_parts = storage.sinks.get();

    for (Sink & s : sink_parts) {
        s.velocity += (dt/2)*s.sph_acceleration;
    }

    for (Sink & s : sink_parts) {
        s.velocity += (dt/2)*s.ext_acceleration;
    }

    for (Sink & s : sink_parts) {
        s.pos += (dt)*s.velocity;
    }

    for (Sink & s : sink_parts) {
        s.velocity += (dt/2)*s.ext_acceleration;
    }

}

template<class Tvec, template<class> class SPHKernel>
void SinkUpdate<Tvec, SPHKernel>::corrector_step(Tscal dt){

    StackEntry stack_loc{};

    if(storage.sinks.is_empty()){
        return;
    }

    
    
    std::vector<Sink> & sink_parts = storage.sinks.get();

    for (Sink & s : sink_parts) {
        s.velocity += (dt/2)*s.sph_acceleration;
    }

}


template<class Tvec, template<class> class SPHKernel>
void SinkUpdate<Tvec, SPHKernel>::compute_sph_forces(Tscal gpart_mass){

    StackEntry stack_loc{};

    if(storage.sinks.is_empty()){
        return;
    }

    std::vector<Sink> & sink_parts = storage.sinks.get();

    Tscal G = solver_config.get_constant_G();
    Tscal epsilon_grav = 1e-9;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 iaxyz_ext      = pdl.get_field_idx<Tvec>("axyz_ext");

    sycl::queue & q = shamsys::instance::get_compute_queue();

    std::vector<Tvec> result_acc_sinks {};

    for (Sink & s : sink_parts) {

        scheduler().for_each_patchdata_nonempty([&,G,epsilon_grav,gpart_mass](Patch cur_p, PatchData &pdat) {

            sycl::buffer<Tvec> &buf_xyz =pdat.get_field_buf_ref<Tvec>(ixyz);
            sycl::buffer<Tvec> &buf_axyz_ext =pdat.get_field_buf_ref<Tvec>(iaxyz_ext);
            
            sycl::buffer<Tvec> buf_sync_axyz (pdat.get_obj_cnt());

            Tscal sink_mass = s.mass;
            Tvec sink_pos = s.pos;
            
            q.submit(
                [&,G,epsilon_grav,sink_mass,sink_pos](sycl::handler &cgh) {

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor axyz_ext{buf_axyz_ext, cgh, sycl::read_write};
                sycl::accessor axyz_sync{buf_sync_axyz, cgh, sycl::write_only, sycl::no_init};

                shambase::parralel_for(cgh, pdat.get_obj_cnt(),"sink-sph forces", [=](i32 id_a){

                    Tvec r_a = xyz[id_a];

                    Tvec delta = r_a - sink_pos;
                    Tscal d = sycl::length(delta);

                    Tvec force = G * delta/(d*d*d);
                    axyz_sync[id_a] = force*gpart_mass;
                    axyz_ext[id_a] += -force*sink_mass;
                    
                });

            });

            result_acc_sinks.push_back(shamalgs::reduction::sum(q, buf_sync_axyz, 0, pdat.get_obj_cnt()));
        });

    }


    std::vector<Tvec> gathered_result_acc_sinks {};
    shamalgs::collective::vector_allgatherv(result_acc_sinks,gathered_result_acc_sinks,MPI_COMM_WORLD);

    u32 id_s = 0;
    for (Sink & s : sink_parts) {

        s.sph_acceleration = {};

        for(u32 rid = 0 ; rid < shamsys::instance::world_size; rid++){
            s.sph_acceleration += gathered_result_acc_sinks[rid*sink_parts.size() + id_s];
        }

        id_s ++;
    }

}


template<class Tvec, template<class> class SPHKernel>
void SinkUpdate<Tvec, SPHKernel>::compute_ext_forces(){

    StackEntry stack_loc{};

    if(storage.sinks.is_empty()){
        return;
    }

    std::vector<Sink> & sink_parts = storage.sinks.get();

    for (Sink & s : sink_parts) {
        s.ext_acceleration = Tvec{};
    }

    Tscal G = solver_config.get_constant_G();
    Tscal epsilon_grav_sink = 1e-9;

    for (Sink & s1 : sink_parts) {
        Tvec sum {};
        for (Sink & s2 : sink_parts) {
            Tvec rij = s1.pos - s2.pos;
            Tscal rij_scal = sycl::length(rij);
            sum -= G*s2.mass*rij/(rij_scal*rij_scal*rij_scal + epsilon_grav_sink);
        }
        s1.ext_acceleration = sum;
    }

}


using namespace shamrock::sph::kernels;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
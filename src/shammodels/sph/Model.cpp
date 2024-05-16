// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Model.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "Model.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"

#include "shamcomm/logs.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/generic/setup/generators.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

#include <functional>
#include <random>
#include <utility>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
using Model = shammodels::sph::Model<Tvec, SPHKernel>;

template<class Tvec, template<class> class SPHKernel>
f64 Model<Tvec, SPHKernel>::evolve_once_time_expl(f64 t_curr, f64 dt_input) {
    auto tmp = solver.evolve_once_time_expl(t_curr, dt_input);
    solver.print_timestep_logs();
    return tmp;
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::timestep() {
    solver.evolve_once();
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::init_scheduler(u32 crit_split, u32 crit_merge) {
    solver.init_required_fields();
    solver.init_ghost_layout();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.add_root_patch();

    logger::debug_ln("Sys", "build local scheduler tables");
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
}

template<class Tvec, template<class> class SPHKernel>
u64 Model<Tvec, SPHKernel>::get_total_part_count() {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    return shamalgs::collective::allreduce_sum(sched.get_rank_count());
}

template<class Tvec, template<class> class SPHKernel>
f64 Model<Tvec, SPHKernel>::total_mass_to_part_mass(f64 totmass) {
    return totmass / get_total_part_count();
}

template<class Tvec, template<class> class SPHKernel>
auto Model<Tvec, SPHKernel>::get_closest_part_to(Tvec pos) -> Tvec {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    Tvec best_dr     = shambase::VectorProperties<Tvec>::get_max();
    Tscal best_dist2 = shambase::VectorProperties<Tscal>::get_max();

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.for_each_patchdata_nonempty([&](const Patch, PatchData &pdat) {
        sycl::buffer<Tvec> &xyz = shambase::get_check_ref(pdat.get_field<Tvec>(0).get_buf());

        sycl::host_accessor acc{xyz, sycl::read_only};

        u32 cnt = pdat.get_obj_cnt();

        for (u32 i = 0; i < cnt; i++) {
            Tvec tmp    = acc[i];
            Tvec dr     = tmp - pos;
            Tscal dist2 = sycl::dot(dr, dr);
            if (dist2 < best_dist2) {
                best_dr    = dr;
                best_dist2 = dist2;
            }
        }
    });

    std::vector<Tvec> list_dr{};
    shamalgs::collective::vector_allgatherv(std::vector<Tvec>{best_dr}, list_dr, MPI_COMM_WORLD);

    // reset distances because if two rank find the same distance the return value won't be the same
    // this bug took me a whole day to fix, aaaaaaaaaaaaah !!!!!
    // maybe this should be moved somewhere else to prevent similar issues
    // TODO (in a year maybe XD )
    best_dr    = shambase::VectorProperties<Tvec>::get_max();
    best_dist2 = shambase::VectorProperties<Tscal>::get_max();

    for (Tvec tmp : list_dr) {
        Tvec dr     = tmp - pos;
        Tscal dist2 = sycl::dot(dr, dr);
        if (dist2 < best_dist2) {
            best_dr    = dr;
            best_dist2 = dist2;
        }
    }

    return pos + best_dr;
}

template<class Tvec, template<class> class SPHKernel>
auto Model<Tvec, SPHKernel>::get_ideal_fcc_box(Tscal dr, std::pair<Tvec, Tvec> box)
    -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};
    auto [a, b] = generic::setup::generators::get_ideal_fcc_box<Tscal>(dr, box);
    return {a, b};
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::remap_positions(std::function<Tvec(Tvec)> map) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    sched.for_each_patchdata_nonempty([&](const Patch, PatchData &pdat) {
        sycl::buffer<Tvec> &xyz = shambase::get_check_ref(pdat.get_field<Tvec>(0).get_buf());

        sycl::host_accessor acc{xyz, sycl::read_write};

        u32 cnt = pdat.get_obj_cnt();

        for (u32 i = 0; i < cnt; i++) {
            acc[i] = map(acc[i]);
        }
    });


    modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
        .update_load_balancing();
    sched.scheduler_step(false, false);

    {
        StackEntry stack_loc{};
        SerialPatchTree<Tvec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<Tvec>());
        shamrock::ReattributeDataUtility reatrib(sched);
        sptree.attach_buf();
        reatrib.reatribute_patch_objects(sptree, "xyz");
        sched.check_patchdata_locality_corectness();
    }

    modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
        .update_load_balancing();
    sched.scheduler_step(true, true);

    {
        StackEntry stack_loc{};
        SerialPatchTree<Tvec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<Tvec>());

        shamrock::ReattributeDataUtility reatrib(sched);
        sptree.attach_buf();
        reatrib.reatribute_patch_objects(sptree, "xyz");
        sched.check_patchdata_locality_corectness();
    }
}

template<class Tvec>
inline void post_insert_data(PatchScheduler &sched) {
    StackEntry stack_loc{};

    // logger::raw_ln(sched.dump_status());
    sched.scheduler_step(false, false);

    /*
            if(shamcomm::world_rank() == 7){
                logger::raw_ln(sched.dump_status());
            }
    */

    auto [m, M] = sched.get_box_tranform<Tvec>();

    {
        StackEntry stack_loc{};
        SerialPatchTree<Tvec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<Tvec>());
        shamrock::ReattributeDataUtility reatrib(sched);
        sptree.attach_buf();
        reatrib.reatribute_patch_objects(sptree, "xyz");
        sched.check_patchdata_locality_corectness();
    }

    sched.scheduler_step(true, true);

    {
        StackEntry stack_loc{};
        SerialPatchTree<Tvec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<Tvec>());

        shamrock::ReattributeDataUtility reatrib(sched);
        sptree.attach_buf();
        reatrib.reatribute_patch_objects(sptree, "xyz");
        sched.check_patchdata_locality_corectness();
    }

    std::string log = "";

    using namespace shamrock::patch;

    u32 smallest_count = u32_max;
    u32 largest_count = 0;

    sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
        u32 tmp = pdat.get_obj_cnt();
        smallest_count = sham::min(tmp, smallest_count);
        largest_count = sham::max(tmp, largest_count);
    });

    smallest_count = shamalgs::collective::allreduce_min(smallest_count);
    largest_count = shamalgs::collective::allreduce_max(largest_count);

    if (shamcomm::world_rank() == 0){
        logger::info_ln("Model", "current particle counts : min = ", smallest_count, "max = ",largest_count);
    }

    //sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
    //    log += shambase::format(
    //        "\n    patch id={}, N={} particles", p.id_patch, pdat.get_obj_cnt());
    //});
    //
    //std::string log_gathered = "";
    //shamcomm::gather_str(log, log_gathered);
    //
    //if (shamcomm::world_rank() == 0)
    //    logger::info_ln("Model", "current particle counts : ", log_gathered);
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::push_particle(
    std::vector<Tvec> &part_pos_insert,
    std::vector<Tscal> &part_hpart_insert,
    std::vector<Tscal> &part_u_insert) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
        PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

        shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

        std::vector<Tvec> vec_acc;
        std::vector<Tscal> hpart_acc;
        std::vector<Tscal> u_acc;
        for (u32 i = 0; i < part_pos_insert.size(); i++) {
            Tvec r  = part_pos_insert[i];
            Tscal u = part_u_insert[i];
            if (patch_coord.contain_pos(r)) {
                vec_acc.push_back(r);
                hpart_acc.push_back(part_hpart_insert[i]);
                u_acc.push_back(u);
            }
        }

        if (vec_acc.size() == 0) {
            return;
        }

        log += shambase::format(
            "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
            shamcomm::world_rank(),
            p.id_patch,
            vec_acc.size(),
            patch_coord.lower,
            patch_coord.upper);

        PatchData tmp(sched.pdl);
        tmp.resize(vec_acc.size());
        tmp.fields_raz();

        {
            u32 len                 = vec_acc.size();
            PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
            sycl::buffer<Tvec> buf(vec_acc.data(), len);
            f.override(buf, len);
        }

        {
            u32 len = vec_acc.size();
            PatchDataField<Tscal> &f
                = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
            sycl::buffer<Tscal> buf(hpart_acc.data(), len);
            f.override(buf, len);
        }

        {
            u32 len                  = u_acc.size();
            PatchDataField<Tscal> &f = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("uint"));
            sycl::buffer<Tscal> buf(u_acc.data(), len);
            f.override(buf, len);
        }

        pdat.insert_elements(tmp);

        sched.check_patchdata_locality_corectness();

        std::string log_gathered = "";
        shamcomm::gather_str(log, log_gathered);

        if (shamcomm::world_rank() == 0) {
            logger::info_ln("Model", "Push particles : ", log_gathered);
        }
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
            .update_load_balancing();

        post_insert_data<Tvec>(sched);
    });
}

template<class Tvec, template<class> class SPHKernel>
auto Model<Tvec, SPHKernel>::get_ideal_hcp_box(Tscal dr, std::pair<Tvec, Tvec> _box)
    -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};

    using Lattice     = shammath::LatticeHCP<Tvec>;
    using LatticeIter = typename shammath::LatticeHCP<Tvec>::Iterator;

    shammath::CoordRange<Tvec> box = _box;
    auto [idxs_min, idxs_max]      = Lattice::get_box_index_bounds(dr, box.lower, box.upper);

    auto [idxs_min_per, idxs_max_per] = Lattice::nearest_periodic_box_indices(idxs_min, idxs_max);

    shammath::CoordRange<Tvec> ret = Lattice::get_periodic_box(dr, idxs_min_per, idxs_max_per);

    return {ret.lower, ret.upper};
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::add_cube_hcp_3d(Tscal dr, std::pair<Tvec, Tvec> _box) {
    shambase::Timer time_setup;time_setup.start();

    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    using Lattice     = shammath::LatticeHCP<Tvec>;
    using LatticeIter = typename shammath::LatticeHCP<Tvec>::IteratorDiscontinuous;

    auto [idxs_min, idxs_max] = Lattice::get_box_index_bounds(dr, box.lower, box.upper);

    LatticeIter gen = LatticeIter(dr, idxs_min, idxs_max);

    u64 acc_count =0;

    std::string log = "";
    while (!gen.is_done()) {

        // loc maximum count of insert part
        u64 loc_sum_ins_cnt = 0;
        // sum_node( loc_sum_ins_cnt )
        u64 max_loc_sum_ins_cnt = 0;

        do {
            std::vector<Tvec> to_ins = gen.next_n(sched.crit_patch_split*2);
            acc_count += to_ins.size();


            sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

                std::vector<Tvec> vec_acc;
                for (Tvec r : to_ins) {
                    if (patch_coord.contain_pos(r)) {
                        vec_acc.push_back(r);
                    }
                }

                // update max insert_count
                loc_sum_ins_cnt += vec_acc.size();

                if (vec_acc.size() == 0) {
                    return;
                }

                log += shambase::format(
                    "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                    shamcomm::world_rank(),
                    p.id_patch,
                    vec_acc.size(),
                    patch_coord.lower,
                    patch_coord.upper);
                
                // reserve space to avoid allocating during copy
                pdat.reserve(vec_acc.size());

                PatchData tmp(sched.pdl);
                tmp.resize(vec_acc.size());
                tmp.fields_raz();

                {
                    u32 len                 = vec_acc.size();
                    PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                    //sycl::buffer<Tvec> buf(vec_acc.data(), len);
                    f.override(vec_acc, len);
                }

                {
                    PatchDataField<Tscal> &f
                        = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                    f.override(dr);
                }

                pdat.insert_elements(tmp);

            });

            max_loc_sum_ins_cnt = shamalgs::collective::allreduce_max(loc_sum_ins_cnt);

            if(shamcomm::world_rank() == 0){
                logger::info_ln("Model", "--> insertion loop : max loc insert count = ",max_loc_sum_ins_cnt, "sum =",acc_count);
            }
        }while(!gen.is_done() && max_loc_sum_ins_cnt < sched.crit_patch_split*8);
        

        sched.check_patchdata_locality_corectness();

        //if(logger::details::loglevel >= shamcomm::logs::log_info){
        //    std::string log_gathered = "";
        //    shamcomm::gather_str(log, log_gathered);
        //
        //    if (shamcomm::world_rank() == 0) {
        //        logger::debug_ln("Model", "Push particles : ", log_gathered);
        //    }
        //}
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
            .update_load_balancing();
        post_insert_data<Tvec>(sched);
        
    }

    if(true){
        modules::ParticleReordering<Tvec,u32, SPHKernel>(ctx, solver.solver_config, solver.storage).reorder_particles();
    }

    time_setup.end();
    if(shamcomm::world_rank() == 0){
        logger::info_ln("Model", "add_cube_hcp took :",time_setup.elasped_sec(),"s");
    }
}

template<class Tvec>
class BigDiscUtils {
public:
using Tscal = shambase::VecComponent<Tvec>;
using Out = generic::setup::generators::DiscOutput<Tscal>; 

class DiscIterator {       
    bool done = false;
    Tvec center;
    Tscal central_mass;
    u64 Npart;
    Tscal r_in;
    Tscal r_out;
    Tscal disc_mass;
    Tscal p;
    Tscal H_r_in;
    Tscal q;
    Tscal G;

    bool do_warp;
    Tscal incl;
    Tscal posangle;
    Tscal Rwarp;
    Tscal Hwarp;

    u64 current_index;
    
    std::mt19937 eng;

    std::function<Tscal(Tscal)> sigma_profile;
    std::function<Tscal(Tscal)> cs_profile;
    std::function<Tscal(Tscal)> rot_profile;

    public:
    DiscIterator(
        Tvec center, 
        Tscal central_mass,
        u64 Npart,
        Tscal r_in,
        Tscal r_out,
        Tscal disc_mass,
        Tscal p,
        Tscal H_r_in,
        Tscal q,
        Tscal G,
        std::mt19937 eng,
        bool do_warp,
        Tscal incl,
        Tscal posangle,
        Tscal Rwarp,
        Tscal Hwarp,
        std::function<Tscal(Tscal)> sigma_profile,
        std::function<Tscal(Tscal)> cs_profile,
        std::function<Tscal(Tscal)> rot_profile
    ) : current_index(0), Npart(Npart),
    center(center), central_mass(central_mass), r_in(r_in), r_out(r_out), disc_mass(disc_mass),
    p(p), H_r_in(H_r_in), q(q), G(G),
    do_warp(do_warp), incl(incl), Rwarp(Rwarp), Hwarp(Hwarp),
    eng(eng), sigma_profile(sigma_profile), cs_profile(cs_profile), rot_profile(rot_profile)
    {

        if (Npart == 0) {
            done = true;
        }
    }

    inline bool is_done() { return done; } // just to make sure the result is not tempered with


    inline generic::setup::generators::DiscOutput<Tscal> next() {

        constexpr Tscal _2pi = 2*shambase::constants::pi<Tscal>;

        auto f_func = [&](Tscal r){
            return r*sigma_profile(r);
        };

        Tscal fmax = f_func(r_out);

        auto find_r = [&](){
            while(true){
                Tscal u2 = shamalgs::random::mock_value<Tscal>(eng,0, fmax);
                Tscal r = shamalgs::random::mock_value<Tscal>(eng,r_in, r_out);
                if (u2 < f_func(r)){
                    return r;
                }
            }
        };
            
        auto theta = shamalgs::random::mock_value<Tscal>(eng,0, _2pi);
        auto Gauss = shamalgs::random::mock_gaussian<Tscal>(eng);

        Tscal r = find_r();

        Tscal vtheta = rot_profile(r);
        Tscal cs = cs_profile(r);
        Tscal sigma = sigma_profile(r);

        Tscal Omega_Kep = sycl::sqrt(G * central_mass/ (r * r * r));
        Tscal H = cs / Omega_Kep;  //sycl::sqrt(2.) * 3. *                 
        Tscal z = H*Gauss;

        auto pos = sycl::vec<Tscal, 3>{r*sycl::cos(theta),r*sycl::sin(theta), z};
        auto vel = sycl::vec<Tscal, 3>{-vtheta*sycl::sin(theta), vtheta*sycl::cos(theta), 0.}; //vk*etheta;

        if (do_warp){
            Tvec k = Tvec(-std::sin(posangle), std::cos(posangle), 0.);
            Tscal inc;
            Tscal psi = 0.;

            //convert to radians (sycl functions take radians)
            Tscal incl_rad = incl * shambase::constants::pi<Tscal> / 180.;
            //Tvec w = sycl::cross(k, pos);
            //Tvec wv = sycl::cross(k, vel);
            //Tscal x = pos.x();
            //Tscal y = pos.y();
            //Tscal z = pos.z();
            //pos = Tvec(x*sycl::cos(incl_rad)-y*sycl::sin(incl_rad), x*sycl::sin(incl_rad)+y*sycl::cos(incl_rad), 0.);
            //logger::info_ln("Model", "LALALALALALALA AVANT LA POS");
            //pos = pos * sycl::cos(inc) + w * sycl::sin(incl_rad) + k * sycl::dot(k, pos) * (1. - sycl::cos(incl_rad));
            //logger::info_ln("Model", "LALALALALALALA APRES LA POS");
            //vel = vel * sycl::cos(inc) + wv * sycl::sin(incl_rad) + k * sycl::dot(k, vel) * (1. - sycl::cos(incl_rad));
            if (r < Rwarp - Hwarp){
                inc = 0.;
            }
            else if (r < Rwarp + 3. * Hwarp && r > Rwarp - Hwarp) {
                inc = sycl::asin(0.5 * (1. + sycl::sin(shambase::constants::pi<Tscal> / (2. * Hwarp) * (r - Rwarp))) * sycl::sin(incl_rad));
                psi = shambase::constants::pi<Tscal> * Rwarp / (4. * Hwarp) * sycl::sin(incl_rad) / sycl::sqrt(1. - (0.5 * sycl::pow(sycl::sin(incl_rad), 2)));
                Tscal psimax = sycl::max(psimax, psi);
                Tscal x = pos.x();
                Tscal y = pos.y();
                Tscal z = pos.z();
                Tvec kk = Tvec(0., 0., 1.);
                Tvec w = sycl::cross(k, pos);
                Tvec wv = sycl::cross(k, vel);
                // Rodrigues' rotation formula
                pos = pos * sycl::cos(inc) + w * sycl::sin(inc) + k * sycl::dot(k, pos) * (1. - sycl::cos(inc));
                vel = vel * sycl::cos(inc) + wv * sycl::sin(inc) + k * sycl::dot(k, vel) * (1. - sycl::cos(inc));

            }
            else{
                inc = 0.;
            }            
        };
        
        Tscal fs = 1.;//1. - sycl::sqrt(r_in / r);
        Tscal rho = (sigma * fs) / (sycl::sqrt(_2pi) * H) * sycl::exp(- z*z / (2*H*H));

        Out out {
            pos, vel, cs, rho
        };

        // increase counter + check if finished
        current_index++;
        if(current_index == Npart){
            done = true;
        }

        return out;
    }

    inline std::vector<Out> next_n(u32 nmax) {
        std::vector<Out> ret{};
        for (u32 i = 0; i < nmax; i++) {
            if (done) {
                break;
            }

            ret.push_back(next());
        }
        return ret;
    }
};

};

            
template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::add_big_disc_3d(
                Tvec center, 
                Tscal central_mass,
                u32 Npart,
                Tscal r_in,
                Tscal r_out,
                Tscal disc_mass,
                Tscal p,
                Tscal H_r_in,
                Tscal q,
                std::mt19937 eng,

                bool do_warp,
                Tscal incl,
                Tscal posangle,
                Tscal Rwarp,
                Tscal Hwarp
) {

    Tscal eos_gamma;

    using Config = SolverConfig;
    using SolverConfigEOS     = typename Config::EOSConfig;
    using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
    if (SolverEOS_Adiabatic *eos_config =
        std::get_if<SolverEOS_Adiabatic>(&solver.solver_config.eos_config.config)) {

        eos_gamma = eos_config->gamma;

    } else {
        //dirty hack for disc setup in locally isothermal
        eos_gamma = 2;
        //shambase::throw_unimplemented();
    }

    auto sigma_profile = [=](Tscal r){
        // we setup with an adimensional mass since it is monte carlo
        constexpr Tscal sigma_0 = 1;
        return sigma_0*sycl::pow(r/r_in, -p);
    };

    auto cs_law = [=](Tscal r){
        return sycl::pow(r/r_in, -q);
    };

//    auto rot_profile = [&](Tscal r){
//        Tscal G = solver.solver_config.get_constant_G();
//        return sycl::sqrt(G * central_mass/r);
//    };

    auto rot_profile2 = [&] (Tscal r) -> Tscal{
        // corrected vphi profile
        //carefull: needs r in cylindrical
        Tscal G = solver.solver_config.get_constant_G();
        Tscal c = solver.solver_config.get_constant_c();
        Tscal aspin = 0.9;
        Tscal term = G * central_mass / r;
        Tscal term_fs = 1. - sycl::sqrt(r_in / r);
        Tscal term_pr = - sycl::pow(cs_law(r), 2) * (1.5 + p + q); // NO CORRECTION from fs term, bad response
        Tscal term_bh = - (2. * aspin / sycl::pow(c, 3)) * sycl::pow(G * central_mass / r, 2);
        Tscal det = sycl::pow(term_bh, 2) + 4.*(term + term_pr);
        Tscal Rg   = G * central_mass / sycl::pow(c, 2);
        Tscal vkep = sqrt(G * central_mass / r);

        Tscal vphi = 0.5*(term_bh + sycl::sqrt(det));

        return vphi;

    };

    auto rot_profile = [&] (Tscal r) -> Tscal{
        // profile for LT
        //carefull: needs r in cylindrical
        Tscal G = solver.solver_config.get_constant_G();
        Tscal c = solver.solver_config.get_constant_c();
        Tscal aspin = 0.9;
        Tscal inc = 30. * shambase::constants::pi<f64> / 180.;
        Tscal Rg   = G * central_mass / sycl::pow(c, 2);
        Tscal vkep = sqrt(G * central_mass / r);
        Tscal r_on_rg = r / Rg;

        Tscal vphi = (sycl::pow(vkep, 4) / (c*c*c)) * (sycl::sqrt(aspin*aspin + sycl::pow(r_on_rg, 3)) - aspin) * sycl::cos(inc);

        return vphi;

    };

    auto cs_profile = [&](Tscal r){
        Tscal cs_in = H_r_in*rot_profile(r_in);
        return cs_law(r)*cs_in; 
    //    return Tscal(0.); //for LT test
    };

    auto get_hfact = []() -> Tscal {
        return Kernel::hfactd;
    };
        
    auto int_rho_h = [&] (Tscal h) -> Tscal{
        return shamrock::sph::rho_h(solver.solver_config.gpart_mass, h, Kernel::hfactd);
    };

    Tscal part_mass = disc_mass/Npart;

    shambase::Timer time_setup;time_setup.start();

    StackEntry stack_loc{};

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    using Out = generic::setup::generators::DiscOutput<Tscal>; 
    using DIter = typename BigDiscUtils<Tvec>::DiscIterator;

    Tscal G = solver.solver_config.get_constant_G();

    DIter gen = DIter(
        center, 
        central_mass,
        Npart,
        r_in,
        r_out,
        disc_mass,
        p,
        H_r_in,
        q,
        G,
        eng,
        do_warp,
        incl,
        posangle,
        Rwarp,
        Hwarp,
        sigma_profile,
        cs_profile,
        rot_profile);

    u64 acc_count =0;

    std::string log = "";
    while (!gen.is_done()) {

        // loc maximum count of insert part
        u64 loc_sum_ins_cnt = 0;
        // sum_node( loc_sum_ins_cnt )
        u64 max_loc_sum_ins_cnt = 0;

        do {
            std::vector<Out> to_ins = gen.next_n(sched.crit_patch_split*2);
            acc_count += to_ins.size();


            sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
                PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

                shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

                std::vector<Out> part_list;
                for (Out r : to_ins) {
                    if (patch_coord.contain_pos(r.pos )) {
                    // add all part to insert in a vector
                    part_list.push_back(r);
                    }
                    else{
                    logger::raw_ln("excluding :", r.pos, "not in :", patch_coord.lower, patch_coord.upper);
                    }
                }

                // update max insert_count
                loc_sum_ins_cnt += part_list.size();

                if (part_list.size() == 0) {
                    return;
                }

                log += shambase::format(
                    "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                    shamcomm::world_rank(),
                    p.id_patch,
                    part_list.size(),
                    patch_coord.lower,
                    patch_coord.upper);
                
                //extract the pos from part_list
                std::vector<Tvec> vec_pos;
                std::vector<Tvec> vec_vel;
                std::vector<Tscal> vec_u;
                std::vector<Tscal> vec_h;
                std::vector<Tscal> vec_cs;

                for(Out o : part_list){
                    vec_pos.push_back(o.pos);
                    vec_vel.push_back(o.velocity);
                    vec_u.push_back(o.cs*o.cs/(/*solver.eos_gamma * */ (eos_gamma - 1)));
                    vec_h.push_back(shamrock::sph::h_rho(part_mass, o.rho * 0.1, Kernel::hfactd));
                    vec_cs.push_back(o.cs);
                }

                // reserve space to avoid allocating during copy
                pdat.reserve(vec_pos.size());

                PatchData tmp(sched.pdl);
                tmp.resize(vec_pos.size());
                tmp.fields_raz();

                {
                    u32 len                 = vec_pos.size();
                    PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                    sycl::buffer<Tvec> buf(vec_pos.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                    sycl::buffer<Tscal> buf(vec_h.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("uint"));
                    sycl::buffer<Tscal> buf(vec_u.data(), len);
                    f.override(buf, len);
                }

                if(solver.solver_config.is_eos_locally_isothermal()){
                    u32 len = vec_pos.size();
                    PatchDataField<Tscal> &f =
                        tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("soundspeed"));
                    sycl::buffer<Tscal> buf(vec_cs.data(), len);
                    f.override(buf, len);
                }

                {
                    u32 len = vec_pos.size();
                    PatchDataField<Tvec> &f =
                        tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
                    sycl::buffer<Tvec> buf(vec_vel.data(), len);
                    f.override(buf, len);
                }

                pdat.insert_elements(tmp);

            });

            max_loc_sum_ins_cnt = shamalgs::collective::allreduce_max(loc_sum_ins_cnt);

            if(shamcomm::world_rank() == 0){
                logger::info_ln("Model", "--> insertion loop : max loc insert count = ",max_loc_sum_ins_cnt, "sum =",acc_count);
            }
        }while(!gen.is_done() && max_loc_sum_ins_cnt < sched.crit_patch_split*8);
        

        sched.check_patchdata_locality_corectness();

        //if(logger::details::loglevel >= shamcomm::logs::log_info){
        //    std::string log_gathered = "";
        //    shamcomm::gather_str(log, log_gathered);
        //
        //    if (shamcomm::world_rank() == 0) {
        //        logger::debug_ln("Model", "Push particles : ", log_gathered);
        //    }
        //}
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
            .update_load_balancing();
        post_insert_data<Tvec>(sched);
        
    }

    if(true){
        modules::ParticleReordering<Tvec,u32, SPHKernel>(ctx, solver.solver_config, solver.storage).reorder_particles();
    }

    time_setup.end();
    if(shamcomm::world_rank() == 0){
        logger::info_ln("Model", "add_big_disc took :",time_setup.elasped_sec(),"s");
    }
}
        
template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::add_cube_fcc_3d(Tscal dr, std::pair<Tvec, Tvec> _box) {
    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    auto make_sliced = [&]() {
        std::vector<Tvec> vec_lst;
        generic::setup::generators::add_particles_fcc(
            dr,
            {box.lower, box.upper},
            [&](Tvec r) {
                return box.contain_pos(r);
            },
            [&](Tvec r, Tscal h) {
                vec_lst.push_back(r);
            });

        std::vector<std::vector<Tvec>> sliced_buf;

        u32 sz_buf = sched.crit_patch_split * 4;

        std::vector<Tvec> cur_buf;
        for (u32 i = 0; i < vec_lst.size(); i++) {
            cur_buf.push_back(vec_lst[i]);

            if (cur_buf.size() > sz_buf) {
                sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
            }
        }

        if (cur_buf.size() > 0) {
            sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
        }

        return sliced_buf;
    };

    std::vector<std::vector<Tvec>> sliced_buf = make_sliced();

    for (std::vector<Tvec> to_ins : sliced_buf) {

        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

            shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

            std::vector<Tvec> vec_acc;
            for (Tvec r : to_ins) {
                if (patch_coord.contain_pos(r)) {
                    vec_acc.push_back(r);
                }
            }

            if (vec_acc.size() == 0) {
                return;
            }

            log += shambase::format(
                "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                shamcomm::world_rank(),
                p.id_patch,
                vec_acc.size(),
                patch_coord.lower,
                patch_coord.upper);

            PatchData tmp(sched.pdl);
            tmp.resize(vec_acc.size());
            tmp.fields_raz();

            {
                u32 len                 = vec_acc.size();
                PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(vec_acc.data(), len);
                f.override(buf, len);
            }

            {
                PatchDataField<Tscal> &f
                    = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                f.override(dr);
            }

            pdat.insert_elements(tmp);
        });

        sched.check_patchdata_locality_corectness();

        std::string log_gathered = "";
        shamcomm::gather_str(log, log_gathered);

        if (shamcomm::world_rank() == 0) {
            logger::info_ln("Model", "Push particles : ", log_gathered);
        }
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
            .update_load_balancing();
        post_insert_data<Tvec>(sched);
    }
}

template<class Tvec, template<class> class SPHKernel>
auto Model<Tvec, SPHKernel>::gen_config_from_phantom_dump(PhantomDump &phdump, bool bypass_error)
    -> SolverConfig {
    StackEntry stack_loc{};
    SolverConfig conf{};

    auto massoftype = phdump.read_header_floats<Tscal>("massoftype");

    conf.gpart_mass = massoftype[0];
    conf.cfl_cour   = phdump.read_header_float<Tscal>("C_force");
    conf.cfl_force  = phdump.read_header_float<Tscal>("C_cour");

    conf.eos_config      = get_shamrock_eosconfig<Tvec>(phdump, bypass_error);
    conf.artif_viscosity = get_shamrock_avconfig<Tvec>(phdump);

    conf.set_units(get_shamrock_units<Tscal>(phdump));

    // xmin, xmax, y... z... are in the header only in periodic mode in phantom
    if (phdump.has_header_entry("xmin")) {
        conf.set_boundary_free();
    }

    return conf;
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::init_from_phantom_dump(PhantomDump &phdump) {
    StackEntry stack_loc{};

    bool has_coord_in_header = true;

    Tscal xmin, xmax, ymin, ymax, zmin, zmax;
    has_coord_in_header = phdump.has_header_entry("xmin");

    std::string log = "";

    std::vector<Tvec> xyz, vxyz;
    std::vector<Tscal> h, u, alpha;

    {
        std::vector<Tscal> x, y, z, vx, vy, vz;

        phdump.blocks[0].fill_vec("x", x);
        phdump.blocks[0].fill_vec("y", y);
        phdump.blocks[0].fill_vec("z", z);

        if (has_coord_in_header) {
            xmin = phdump.read_header_float<f64>("xmin");
            xmax = phdump.read_header_float<f64>("xmax");
            ymin = phdump.read_header_float<f64>("ymin");
            ymax = phdump.read_header_float<f64>("ymax");
            zmin = phdump.read_header_float<f64>("zmin");
            zmax = phdump.read_header_float<f64>("zmax");

            resize_simulation_box({{xmin, ymin, zmin}, {xmax, ymax, zmax}});
        } else {
            Tscal box_tolerance = 1.2;

            xmin = *std::min_element(x.begin(), x.end());
            xmax = *std::max_element(x.begin(), x.end());
            ymin = *std::min_element(y.begin(), y.end());
            ymax = *std::max_element(y.begin(), y.end());
            zmin = *std::min_element(z.begin(), z.end());
            zmax = *std::max_element(z.begin(), z.end());

            Tvec bm = {xmin, ymin, zmin};
            Tvec bM = {xmax, ymax, zmax};

            Tvec center = (bm + bM) * 0.5;

            Tvec d = (bM - bm) * 0.5;

            // expand the box
            d *= box_tolerance;

            resize_simulation_box({center - d, center + d});
        }

        phdump.blocks[0].fill_vec("h", h);

        phdump.blocks[0].fill_vec("vx", vx);
        phdump.blocks[0].fill_vec("vy", vy);
        phdump.blocks[0].fill_vec("vz", vz);

        phdump.blocks[0].fill_vec("u", u);
        phdump.blocks[0].fill_vec("alpha", alpha);

        for (u32 i = 0; i < x.size(); i++) {
            xyz.push_back({x[i], y[i], z[i]});
        }
        for (u32 i = 0; i < vx.size(); i++) {
            vxyz.push_back({vx[i], vy[i], vz[i]});
        }
    }

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    u32 sz_buf = sched.crit_patch_split * 4;

    u32 Ntot = xyz.size();

    std::vector<u64> insert_ranges;
    insert_ranges.push_back(0);
    for (u64 i = sz_buf; i < Ntot; i += sz_buf) {
        insert_ranges.push_back(i);
    }
    insert_ranges.push_back(Ntot);

    for (u64 krange = 0; krange < insert_ranges.size() - 1; krange++) {
        u64 start_id = insert_ranges[krange];
        u64 end_id   = insert_ranges[krange + 1];

        u64 Nloc = end_id - start_id;

        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

            shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

            std::vector<u64> sel_index;
            for (u64 i = start_id; i < end_id; i++) {
                Tvec r   = xyz[i];
                Tscal h_ = h[i];
                if (patch_coord.contain_pos(r) && (h_ >= 0)) {
                    sel_index.push_back(i);
                }
            }

            if (sel_index.size() == 0) {
                return;
            }

            log += shambase::format(
                "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                shamcomm::world_rank(),
                p.id_patch,
                sel_index.size(),
                patch_coord.lower,
                patch_coord.upper);

            std::vector<Tvec> ins_xyz, ins_vxyz;
            std::vector<Tscal> ins_h, ins_u, ins_alpha;
            for (u64 i : sel_index) {
                ins_xyz.push_back(xyz[i]);
            }
            for (u64 i : sel_index) {
                ins_vxyz.push_back(vxyz[i]);
            }
            for (u64 i : sel_index) {
                ins_h.push_back(h[i]);
            }
            if (u.size() > 0) {
                for (u64 i : sel_index) {
                    ins_u.push_back(u[i]);
                }
            }
            if (alpha.size() > 0) {
                for (u64 i : sel_index) {
                    ins_alpha.push_back(alpha[i]);
                }
            }

            PatchData ptmp(sched.pdl);
            ptmp.resize(sel_index.size());
            ptmp.fields_raz();

            ptmp.override_patch_field("xyz", ins_xyz);
            ptmp.override_patch_field("vxyz", ins_vxyz);
            ptmp.override_patch_field("hpart", ins_h);

            if (ins_alpha.size() > 0) {
                ptmp.override_patch_field("alpha_AV", ins_alpha);
            }

            if (ins_u.size() > 0) {
                ptmp.override_patch_field("uint", ins_u);
            }

            pdat.insert_elements(ptmp);
        });

        sched.check_patchdata_locality_corectness();

        std::string log_gathered = "";
        shamcomm::gather_str(log, log_gathered);

        if (shamcomm::world_rank() == 0) {
            logger::info_ln("Model", "Push particles : ", log_gathered);
        }
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel>(ctx, solver.solver_config, solver.storage)
            .update_load_balancing();

        post_insert_data<Tvec>(sched);

        // add sinks

        PhantomDumpBlock &sink_block = phdump.blocks[1];
        {
            std::vector<Tscal> xsink, ysink, zsink;
            std::vector<Tscal> vxsink, vysink, vzsink;
            std::vector<Tscal> mass;
            std::vector<Tscal> Racc;

            sink_block.fill_vec("x", xsink);
            sink_block.fill_vec("y", ysink);
            sink_block.fill_vec("z", zsink);
            sink_block.fill_vec("vx", vxsink);
            sink_block.fill_vec("vy", vysink);
            sink_block.fill_vec("vz", vzsink);
            sink_block.fill_vec("m", mass);
            sink_block.fill_vec("h", Racc);

            for (u32 i = 0; i < xsink.size(); i++) {
                add_sink(
                    mass[i],
                    {xsink[i], ysink[i], zsink[i]},
                    {vxsink[i], vysink[i], vzsink[i]},
                    Racc[i]);
            }
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::add_pdat_to_phantom_block(
    PhantomDumpBlock &block, shamrock::patch::PatchData &pdat) {

    std::vector<Tvec> xyz = pdat.fetch_data<Tvec>("xyz");

    u64 xid = block.get_ref_fort_real("x");
    u64 yid = block.get_ref_fort_real("y");
    u64 zid = block.get_ref_fort_real("z");

    for (auto vec : xyz) {
        block.blocks_fort_real[xid].vals.push_back(vec.x());
        block.blocks_fort_real[yid].vals.push_back(vec.y());
        block.blocks_fort_real[zid].vals.push_back(vec.z());
    }

    std::vector<Tscal> h = pdat.fetch_data<Tscal>("hpart");
    u64 hid              = block.get_ref_f32("h");
    for (auto h_ : h) {
        block.blocks_f32[hid].vals.push_back(h_);
    }

    if (solver.solver_config.has_field_alphaAV()) {
        std::vector<Tscal> alpha = pdat.fetch_data<Tscal>("alpha_AV");
        u64 aid                  = block.get_ref_f32("alpha");
        for (auto alp_ : alpha) {
            block.blocks_f32[aid].vals.push_back(alp_);
        }
    }

    if (solver.solver_config.has_field_divv()) {
        std::vector<Tscal> vecdivv = pdat.fetch_data<Tscal>("divv");
        u64 divvid                 = block.get_ref_f32("divv");
        for (auto d_ : vecdivv) {
            block.blocks_f32[divvid].vals.push_back(d_);
        }
    }

    std::vector<Tvec> vxyz = pdat.fetch_data<Tvec>("vxyz");

    u64 vxid = block.get_ref_fort_real("vx");
    u64 vyid = block.get_ref_fort_real("vy");
    u64 vzid = block.get_ref_fort_real("vz");

    for (auto vec : vxyz) {
        block.blocks_fort_real[vxid].vals.push_back(vec.x());
        block.blocks_fort_real[vyid].vals.push_back(vec.y());
        block.blocks_fort_real[vzid].vals.push_back(vec.z());
    }

    std::vector<Tscal> u = pdat.fetch_data<Tscal>("uint");
    u64 uid              = block.get_ref_fort_real("u");
    for (auto u_ : u) {
        block.blocks_fort_real[uid].vals.push_back(u_);
    }

    block.tot_count = block.blocks_fort_real[xid].vals.size();
}

template<class Tvec, template<class> class SPHKernel>
shammodels::sph::PhantomDump Model<Tvec, SPHKernel>::make_phantom_dump() {
    StackEntry stack_loc{};

    PhantomDump dump;

    dump.override_magic_number();
    dump.iversion = 1;
    dump.fileid   = shambase::format("{:100s}", "FT:Phantom Shamrock writter");

    u32 Ntot = get_total_part_count();
    dump.table_header_fort_int.add("nparttot", Ntot);
    dump.table_header_fort_int.add("ntypes", 8);
    dump.table_header_fort_int.add("npartoftype", Ntot);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);
    dump.table_header_fort_int.add("npartoftype", 0);

    dump.table_header_i64.add("nparttot", Ntot);
    dump.table_header_i64.add("ntypes", 8);
    dump.table_header_i64.add("npartoftype", Ntot);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);
    dump.table_header_i64.add("npartoftype", 0);

    dump.table_header_fort_int.add("nblocks", 1);
    dump.table_header_fort_int.add("nptmass", 0);
    dump.table_header_fort_int.add("ndustlarge", 0);
    dump.table_header_fort_int.add("ndustsmall", 0);
    dump.table_header_fort_int.add("idust", 7);
    dump.table_header_fort_int.add("idtmax_n", 1);
    dump.table_header_fort_int.add("idtmax_frac", 0);
    dump.table_header_fort_int.add("idumpfile", 0);
    dump.table_header_fort_int.add("majorv", 2023);
    dump.table_header_fort_int.add("minorv", 0);
    dump.table_header_fort_int.add("microv", 0);
    dump.table_header_fort_int.add("isink", 0);

    dump.table_header_i32.add("iexternalforce", 0);
    dump.table_header_i32.add("ieos", 2);
    dump.table_header_fort_real.add("gamma", 1.66667);
    dump.table_header_fort_real.add("RK2", 0);
    dump.table_header_fort_real.add("polyk2", 0);
    dump.table_header_fort_real.add("qfacdisc", 0.75);
    dump.table_header_fort_real.add("qfacdisc2", 0.75);

    dump.table_header_fort_real.add("time", solver.solver_config.get_time());
    dump.table_header_fort_real.add("dtmax", solver.solver_config.get_dt_sph());

    dump.table_header_fort_real.add("rhozero", 0);
    dump.table_header_fort_real.add("hfact", Kernel::hfactd);
    dump.table_header_fort_real.add("tolh", 0.0001);
    dump.table_header_fort_real.add("C_cour", solver.solver_config.cfl_cour);
    dump.table_header_fort_real.add("C_force", solver.solver_config.cfl_force);
    dump.table_header_fort_real.add("alpha", 0);
    dump.table_header_fort_real.add("alphau", 1);
    dump.table_header_fort_real.add("alphaB", 1);

    dump.table_header_fort_real.add("massoftype", solver.solver_config.gpart_mass);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);
    dump.table_header_fort_real.add("massoftype", 0);

    dump.table_header_fort_real.add("Bextx", 0);
    dump.table_header_fort_real.add("Bexty", 0);
    dump.table_header_fort_real.add("Bextz", 0);
    dump.table_header_fort_real.add("dum", 0);

    PatchScheduler &sched = shambase::get_check_ref(solver.context.sched);

    auto [bmin, bmax] = sched.get_box_volume<Tvec>();

    dump.table_header_fort_real.add("xmin", bmin.x());
    dump.table_header_fort_real.add("xmax", bmax.x());
    dump.table_header_fort_real.add("ymin", bmin.y());
    dump.table_header_fort_real.add("ymax", bmax.x());
    dump.table_header_fort_real.add("zmin", bmin.z());
    dump.table_header_fort_real.add("zmax", bmax.x());

    dump.table_header_fort_real.add("get_conserv", -1);
    dump.table_header_fort_real.add("etot_in", 0.59762);
    dump.table_header_fort_real.add("angtot_in", 0.0189694);
    dump.table_header_fort_real.add("totmom_in", 0.0306284);

    auto units = solver.solver_config.unit_sys;
    if (units) {
        dump.table_header_f64.add("udist", units->m_inv);
        dump.table_header_f64.add("umass", units->kg_inv);
        dump.table_header_f64.add("utime", units->s_inv);

        f64 umass = units->template to<shamunits::units::kg>();
        f64 utime = units->template to<shamunits::units::s>();
        f64 udist = units->template to<shamunits::units::m>();

        shamunits::Constants<double> ctes{*units};
        f64 ccst    = ctes.c();
        f64 ucharge = sqrt(umass * udist / (4. * shambase::constants::pi<f64> /*mu_0 in cgs*/));

        f64 umagfd = umass / (utime * ucharge);

        dump.table_header_f64.add("umagfd", umagfd);
    } else {
        logger::warn_ln("SPH", "no units are set, defaulting to SI");

        dump.table_header_f64.add("udist", 1);
        dump.table_header_f64.add("umass", 1);
        dump.table_header_f64.add("utime", 1);
        dump.table_header_f64.add("umagfd", 3.54491);
    }

    PhantomDumpBlock block_part;

    {
        NamedStackEntry stack_loc{"gather data"};
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered = ctx.allgather_data();

        for (auto &dat : gathered) {
            add_pdat_to_phantom_block(block_part, shambase::get_check_ref(dat));
        }
    }

    dump.blocks.push_back(std::move(block_part));

    return dump;
}

using namespace shammath;

template class shammodels::sph::Model<f64_3, M4>;
template class shammodels::sph::Model<f64_3, M6>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "Model.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
using Model = shammodels::sph::Model<Tvec, SPHKernel>;

template<class Tvec, template<class> class SPHKernel>
f64 Model<Tvec, SPHKernel>::evolve_once(f64 t_curr, f64 dt_input,
                                                       bool do_dump,
                                                       std::string vtk_dump_name,
                                                       bool vtk_dump_patch_id) {
    return solver.evolve_once(t_curr,dt_input, do_dump, vtk_dump_name, vtk_dump_patch_id);
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
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();
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
auto Model<Tvec, SPHKernel>::get_closest_part_to(Tvec pos) -> Tvec{

    using namespace shamrock::patch;

    Tvec best_dr = shambase::VectorProperties<Tvec>::get_max();
    Tscal best_dist2 = shambase::VectorProperties<Tscal>::get_max();

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.for_each_patchdata_nonempty([&](const Patch, PatchData & pdat){
        sycl::buffer<Tvec> & xyz = shambase::get_check_ref(pdat.get_field<Tvec>(0).get_buf());

        sycl::host_accessor acc {xyz, sycl::read_only};

        u32 cnt = pdat.get_obj_cnt();

        for(u32 i = 0; i < cnt; i++){
            Tvec tmp = acc[i];
            Tvec dr = tmp - pos;
            Tscal dist2 = sycl::dot(dr,dr);
            if(dist2 < best_dist2){
                best_dr = dr;
                best_dist2 = dist2;
            }
        }
    });


    std::vector<Tvec> list_dr {};
    shamalgs::collective::vector_allgatherv(std::vector<Tvec>{best_dr},list_dr,MPI_COMM_WORLD);


    for(Tvec tmp : list_dr){
        Tvec dr = tmp - pos;
        Tscal dist2 = sycl::dot(dr,dr);
        if(dist2 < best_dist2){
            best_dr = dr;
            best_dist2 = dist2;
        }
    }

    return pos + best_dr;

}

using namespace shamrock::sph::kernels;

template class shammodels::sph::Model<f64_3, M4>;
template class shammodels::sph::Model<f64_3, M6>;
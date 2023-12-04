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
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <utility>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
using Model = shammodels::sph::Model<Tvec, SPHKernel>;

template<class Tvec, template<class> class SPHKernel>
f64 Model<Tvec, SPHKernel>::evolve_once(
    f64 t_curr, f64 dt_input, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id) {
    return solver.evolve_once(t_curr, dt_input, do_dump, vtk_dump_name, vtk_dump_patch_id);
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
    sched.update_local_load_value([&](shamrock::patch::Patch p){
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

template<class Tvec>
inline void post_insert_data(PatchScheduler &sched) {
    StackEntry stack_loc{};

    logger::raw_ln(sched.dump_status());
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
    sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
        log +=
            shambase::format("\n    patch id={}, N={} particles", p.id_patch, pdat.get_obj_cnt());
    });

    std::string log_gathered = "";
    shamcomm::gather_str(log, log_gathered);

    if (shamcomm::world_rank() == 0)
        logger::info_ln("Model", "current particle counts : ", log_gathered);
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
            PatchDataField<Tscal> &f =
                tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
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

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();

        post_insert_data<Tvec>(sched);
    });
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
                PatchDataField<Tscal> &f =
                    tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
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
        
        modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();
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

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();

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

/**
 * @brief 
 * \todo move to patchdata
 * @tparam T 
 * @param key 
 * @param pdat 
 * @return std::vector<T> 
 */
template<class T>
std::vector<T> fetch_data(std::string key, shamrock::patch::PatchData & pdat){

    std::vector<T> vec;


    auto appender = [&](auto & field){

        if (field.get_name() == key) {

            logger::debug_ln("PyShamrockCTX","appending field",key);
            
            {
                sycl::host_accessor acc {shambase::get_check_ref(field.get_buf())};
                u32 len = field.size();

                for (u32 i = 0 ; i < len; i++) {
                    vec.push_back(acc[i]);
                }
            }

        }

    };



    pdat.for_each_field<T>([&](auto & field){
        appender(field);
    });

    return vec;

}


template<class Tvec, template<class> class SPHKernel>
void Model<Tvec, SPHKernel>::add_pdat_to_phantom_block(PhantomDumpBlock & block, shamrock::patch::PatchData & pdat){
    
    std::vector<Tvec> xyz = fetch_data<Tvec>("xyz", pdat);

    u64 xid = block.get_ref_fort_real("x");
    u64 yid = block.get_ref_fort_real("y");
    u64 zid = block.get_ref_fort_real("z");

    for (auto vec : xyz) {
        block.blocks_fort_real[xid].vals.push_back(vec.x());
        block.blocks_fort_real[yid].vals.push_back(vec.y());
        block.blocks_fort_real[zid].vals.push_back(vec.z());
    }


    std::vector<Tscal> h = fetch_data<Tscal>("hpart", pdat);
    u64 hid = block.get_ref_f32("h");
    for (auto h_ : h) {
        block.blocks_f32[hid].vals.push_back(h_);
    }


    if(solver.solver_config.has_field_alphaAV()){
        std::vector<Tscal> alpha = fetch_data<Tscal>("alpha_AV", pdat);
        u64 aid = block.get_ref_f32("alpha");
        for (auto alp_ : alpha) {
            block.blocks_f32[aid].vals.push_back(alp_);
        }
    }

    if(solver.solver_config.has_field_divv()){
        std::vector<Tscal> vecdivv = fetch_data<Tscal>("divv", pdat);
        u64 divvid = block.get_ref_f32("divv");
        for (auto d_ : vecdivv) {
            block.blocks_f32[divvid].vals.push_back(d_);
        }
    }



    std::vector<Tvec> vxyz = fetch_data<Tvec>("vxyz", pdat);

    u64 vxid = block.get_ref_fort_real("vx");
    u64 vyid = block.get_ref_fort_real("vy");
    u64 vzid = block.get_ref_fort_real("vz");

    for (auto vec : vxyz) {
        block.blocks_fort_real[vxid].vals.push_back(vec.x());
        block.blocks_fort_real[vyid].vals.push_back(vec.y());
        block.blocks_fort_real[vzid].vals.push_back(vec.z());
    }

    std::vector<Tscal> u = fetch_data<Tscal>("uint", pdat);
    u64 uid = block.get_ref_fort_real("u");
    for (auto u_ : u) {
        block.blocks_fort_real[uid].vals.push_back(u_);
    }

    block.tot_count = block.blocks_fort_real[xid].vals.size();

}

template<class Tvec, template<class> class SPHKernel>
shammodels::sph::PhantomDump Model<Tvec, SPHKernel>::make_phantom_dump() {

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


    dump.table_header_fort_real.add("time", 0);
    dump.table_header_fort_real.add("dtmax", 0.1);


    dump.table_header_fort_real.add("rhozero", 0);
    dump.table_header_fort_real.add("hfact", Kernel::hfactd);
    dump.table_header_fort_real.add("tolh", 0.0001);
    dump.table_header_fort_real.add("C_cour", solver.solver_config.cfl_cour);
    dump.table_header_fort_real.add("C_force",solver.solver_config.cfl_force);
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

    PatchScheduler & sched = shambase::get_check_ref(solver.context.sched);
        
    auto [bmin,bmax] = sched.get_box_volume<Tvec>();

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
        dump.table_header_f64.add("umagfd", 3.54491);
    }else {
        shambase::throw_unimplemented();
    }



    PhantomDumpBlock block_part;
    
    {
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered = ctx.allgather_data();

        for(auto & dat : gathered){
            add_pdat_to_phantom_block(block_part, shambase::get_check_ref(dat));
        }

    }
    
    dump.blocks.push_back(std::move(block_part));





    return dump;
}

using namespace shammath;

template class shammodels::sph::Model<f64_3, M4>;
template class shammodels::sph::Model<f64_3, M6>;

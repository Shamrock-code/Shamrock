// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Model.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief GSPH Model implementation
 */

#include "shambase/aliases_float.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/setup/generators.hpp"
#include "shammodels/gsph/Model.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <functional>
#include <utility>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::init_scheduler(u32 crit_split, u32 crit_merge) {
    solver.init_required_fields();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.add_root_patch();

    shamlog_debug_ln("Sys", "build local scheduler tables");
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
    solver.init_ghost_layout();

    solver.init_solver_graph();
}

template<class Tvec, template<class> class SPHKernel>
u64 shammodels::gsph::Model<Tvec, SPHKernel>::get_total_part_count() {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    return shamalgs::collective::allreduce_sum(sched.get_rank_count());
}

template<class Tvec, template<class> class SPHKernel>
f64 shammodels::gsph::Model<Tvec, SPHKernel>::total_mass_to_part_mass(f64 totmass) {
    return totmass / get_total_part_count();
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::gsph::Model<Tvec, SPHKernel>::get_ideal_fcc_box(
    Tscal dr, std::pair<Tvec, Tvec> box) -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};
    auto [a, b] = generic::setup::generators::get_ideal_fcc_box<Tscal>(
        dr, std::make_tuple(box.first, box.second));
    return {a, b};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::gsph::Model<Tvec, SPHKernel>::get_ideal_hcp_box(
    Tscal dr, std::pair<Tvec, Tvec> box) -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};
    auto [a, b] = generic::setup::generators::get_ideal_fcc_box<Tscal>(
        dr, std::make_tuple(box.first, box.second));
    return {a, b};
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::add_cube_fcc_3d(
    Tscal dr, std::pair<Tvec, Tvec> _box) {
    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    auto make_sliced = [&]() {
        std::vector<Tvec> vec_lst;
        generic::setup::generators::add_particles_fcc(
            dr,
            std::make_tuple(box.lower, box.upper),
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

        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            PatchCoordTransform<Tvec> ptransf
                = sched.get_sim_box().template get_patch_transform<Tvec>();

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

            PatchDataLayer tmp(sched.get_layout_ptr_old());
            tmp.resize(vec_acc.size());
            tmp.fields_raz();

            {
                u32 len                 = vec_acc.size();
                PatchDataField<Tvec> &f = tmp.template get_field<Tvec>(
                    sched.pdl_old().template get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(vec_acc.data(), len);
                f.override(buf, len);
            }

            {
                PatchDataField<Tscal> &f = tmp.template get_field<Tscal>(
                    sched.pdl_old().template get_field_idx<Tscal>("hpart"));
                using Kernel = SPHKernel<Tscal>;
                f.override(Kernel::hfactd * dr);
            }

            pdat.insert_elements(tmp);
        });

        sched.check_patchdata_locality_corectness();
        sched.scheduler_step(true, true);
    }

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    shamlog_debug_ln("setup", log);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::add_cube_hcp_3d(
    Tscal dr, std::pair<Tvec, Tvec> _box) {
    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    auto make_sliced = [&]() {
        std::vector<Tvec> vec_lst;
        generic::setup::generators::add_particles_fcc(
            dr,
            std::make_tuple(box.lower, box.upper),
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

        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            PatchCoordTransform<Tvec> ptransf
                = sched.get_sim_box().template get_patch_transform<Tvec>();

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

            PatchDataLayer tmp(sched.get_layout_ptr_old());
            tmp.resize(vec_acc.size());
            tmp.fields_raz();

            {
                u32 len                 = vec_acc.size();
                PatchDataField<Tvec> &f = tmp.template get_field<Tvec>(
                    sched.pdl_old().template get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(vec_acc.data(), len);
                f.override(buf, len);
            }

            {
                PatchDataField<Tscal> &f = tmp.template get_field<Tscal>(
                    sched.pdl_old().template get_field_idx<Tscal>("hpart"));
                using Kernel = SPHKernel<Tscal>;
                f.override(Kernel::hfactd * dr);
            }

            pdat.insert_elements(tmp);
        });

        sched.check_patchdata_locality_corectness();
        sched.scheduler_step(true, true);
    }

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    shamlog_debug_ln("setup", log);
}

// Explicit template instantiations for all supported kernel types
template class shammodels::gsph::Model<f64_3, shammath::M4>;
template class shammodels::gsph::Model<f64_3, shammath::M6>;
template class shammodels::gsph::Model<f64_3, shammath::M8>;
template class shammodels::gsph::Model<f64_3, shammath::C2>;
template class shammodels::gsph::Model<f64_3, shammath::C4>;
template class shammodels::gsph::Model<f64_3, shammath::C6>;
